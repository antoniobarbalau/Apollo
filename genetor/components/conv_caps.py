import tensorflow as tf
from .basic import to_tensor
from .initializations import default_initialization
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from .convolutions import conv


epsilon = 1e-7


def conv_caps_primary(input, **params):
    n_capsules = params.get('n_capsules', 32)
    pose_shape = params.get('pose_shape', [4, 4])
    padding = params.get('padding', 'VALID')

    with tf.variable_scope(params['name']) as scope:
        poses = conv(
            input,
            filters = n_capsules * pose_shape[0] * pose_shape[1],
            kernel_size = 1,
            stride = 1,
            padding = padding,
            activation = None,
            name = 'poses_conv'
        )
        poses = tf.reshape(
            poses,
            [
                -1,
                input.shape[1].value, input.shape[2].value,
                n_capsules, pose_shape[0], pose_shape[1]
            ],
            name = 'poses'
        )

        activations = conv(
            input,
            filters = n_capsules,
            kernel_size = 1,
            stride = 1,
            padding = padding,
            activation = tf.nn.sigmoid,
            name = 'activations_conv'
        )
        activations = tf.identity(activations, name = 'activations')

    return poses, activations


def conv_caps(input, **params):
  stride = params.get('stride', 2)
  # strides = params.get('strides', [1, 2, 2, 1])
  strides = [1, stride, stride, 1]
  iterations = 2
  batch_size = 10
  inputs_poses, inputs_activations = input
  shape = [
      3, 3, inputs_poses.shape[3].value, params.get('n_capsules', 32)
  ]
  # batch_size = tf.shape(inputs_poses)[0]

  with tf.variable_scope(params['name']) as scope:

      stride = strides[1] # 2
      i_size = shape[-2] # 32
      o_size = shape[-1] # 32
      pose_size = inputs_poses.get_shape()[-1]  # 4

      # Tile the input capusles' pose matrices to the spatial dimension of the output capsules
      # Such that we can later multiple with the transformation matrices to generate the votes.
      inputs_poses = kernel_tile(inputs_poses, 3, stride)  # (?, 14, 14, 32, 4, 4) -> (?, 6, 6, 3x3=9, 32x16=512)

      # Tile the activations needed for the EM routing
      inputs_activations = kernel_tile(inputs_activations, 3, stride)  # (?, 14, 14, 32) -> (?, 6, 6, 9, 32)
      spatial_size = int(inputs_activations.get_shape()[1]) # 6
      spatial_size_2 = int(inputs_activations.get_shape()[2]) # 6

      # Reshape it for later operations
      inputs_poses = tf.reshape(inputs_poses, shape=[-1, 3 * 3 * i_size, 16])  # (?, 9x32=288, 16)
      inputs_activations = tf.reshape(inputs_activations, shape=[-1, spatial_size, spatial_size_2, 3 * 3 * i_size]) # (?, 6, 6, 9x32=288)

      with tf.variable_scope('votes') as scope:
          
          # Generate the votes by multiply it with the transformation matrices
          votes = mat_transform(inputs_poses, o_size, batch_size*spatial_size*spatial_size_2)  # (864, 288, 32, 16)
          
          # Reshape the vote for EM routing
          votes_shape = votes.get_shape()
          votes = tf.reshape(votes, shape=[batch_size, spatial_size, spatial_size_2, votes_shape[-3], votes_shape[-2], votes_shape[-1]]) # (24, 6, 6, 288, 32, 16)
          # tf.logging.info(f"{name} votes shape: {votes.get_shape()}")

      with tf.variable_scope('routing') as scope:
          
          # beta_v and beta_a one for each output capsule: (1, 1, 1, 32)
          beta_v = tf.get_variable(
              name='beta_v', shape=[1, 1, 1, o_size], dtype=tf.float32,
              initializer=initializers.xavier_initializer()
          )
          beta_a = tf.get_variable(
              name='beta_a', shape=[1, 1, 1, o_size], dtype=tf.float32,
              initializer=initializers.xavier_initializer()
          )

          # Use EM routing to compute the pose and activation
          # votes (24, 6, 6, 3x3x32=288, 32, 16), inputs_activations (?, 6, 6, 288)
          # poses (24, 6, 6, 32, 16), activation (24, 6, 6, 32)
          poses, activations = matrix_capsules_em_routing(
              votes, inputs_activations, beta_v, beta_a, iterations, name='em_routing'
          )

          # Reshape it back to 4x4 pose matrix
          poses_shape = poses.get_shape()
          # (24, 6, 6, 32, 4, 4)
          poses = tf.reshape(
              poses, [
                  poses_shape[0], poses_shape[1], poses_shape[2], poses_shape[3], pose_size, pose_size
              ]
          )
          print(poses.shape)

      # tf.logging.info(f"{name} pose shape: {poses.get_shape()}")
      # tf.logging.info(f"{name} activations shape: {activations.get_shape()}")

      return poses, activations


def kernel_tile(input, kernel_size, stride):
    input = tf.reshape(
        input,
        [
            -1,
            input.shape[1].value, input.shape[2].value,
            np.prod(input.shape[3:])
        ]
    )

    tile_kernel = np.zeros(
        [
            kernel_size, kernel_size,
            input.shape[3].value,
            kernel_size * kernel_size
        ],
        dtype = np.float32
    )
    for i in range(kernel_size):
        for j in range(kernel_size):
            tile_kernel[i, j, :, i * kernel_size + j] = 1.
    tile_kernel = tf.constant(tile_kernel, dtype = tf.float32)

    output = tf.nn.depthwise_conv2d(
        input,
        tile_kernel,
        strides = [1, stride, stride, 1],
        padding = 'VALID'
    )
    output = tf.reshape(
        output,
        shape = [
            -1,
            output.shape[1].value, output.shape[2].value,
            input.shape[3].value, kernel_size * kernel_size
        ]
    )
    output = tf.transpose(output, [0, 1, 2, 4, 3])

    return output


def mat_transform(input, n_output_caps, n_caps_tuples):
    n_input_caps = input.shape[1].value

    input = tf.reshape(
        input,
        [-1, n_input_caps, 1, 4, 4]
    )
    input = tf.tile(input, [1, 1, n_output_caps, 1, 1])

    w = tf.Variable(
        initial_value = tf.truncated_normal(
            mean = 0.,
            stddev = 1.,
            shape = [1, n_input_caps, n_output_caps, 4, 4]
        ),
        dtype = tf.float32
    )
    w = tf.tile(w, [n_caps_tuples, 1, 1, 1, 1])
    output = tf.matmul(input, w)
    output = tf.reshape(output, [-1, n_input_caps, n_output_caps, 16])

    return output


def matrix_capsules_em_routing(votes, i_activations, beta_v, beta_a, iterations, name):
  """The EM routing between input capsules (i) and output capsules (j).

  :param votes: (N, OH, OW, kh x kw x i, o, 4 x 4) = (24, 6, 6, 3x3*32=288, 32, 16)
  :param i_activation: activation from Level L (24, 6, 6, 288)
  :param beta_v: (1, 1, 1, 32)
  :param beta_a: (1, 1, 1, 32)
  :param iterations: number of iterations in EM routing, often 3.
  :param name: name.

  :return: (pose, activation) of output capsules.
  """

  votes_shape = votes.get_shape().as_list()

  with tf.variable_scope(name) as scope:

    # Match rr (routing assignment) shape, i_activations shape with votes shape for broadcasting in EM routing

    # rr: [3x3x32=288, 32, 1]
    # rr: routing matrix from each input capsule (i) to each output capsule (o)
    rr = tf.constant(
      1.0/votes_shape[-2], shape=votes_shape[-3:-1] + [1], dtype=tf.float32
    )

    # i_activations: expand_dims to (24, 6, 6, 288, 1, 1)
    i_activations = i_activations[..., tf.newaxis, tf.newaxis]

    # beta_v and beta_a: expand_dims to (1, 1, 1, 1, 32, 1]
    beta_v = beta_v[..., tf.newaxis, :, tf.newaxis]
    beta_a = beta_a[..., tf.newaxis, :, tf.newaxis]

    # inverse_temperature schedule (min, max)
    it_min = 1.0
    it_max = min(iterations, 3.0)
    for it in range(iterations):
      inverse_temperature = it_min + (it_max - it_min) * it / max(1.0, iterations - 1.0)
      o_mean, o_stdv, o_activations = m_step(
        rr, votes, i_activations, beta_v, beta_a, inverse_temperature=inverse_temperature
      )

      # We skip the e_step call in the last iteration because we only 
      # need to return the a_j and the mean from the m_stp in the last iteration
      # to compute the output capsule activation and pose matrices  
      if it < iterations - 1:
        rr = e_step(
          o_mean, o_stdv, o_activations, votes
        )

    # pose: (N, OH, OW, o 4 x 4) via squeeze o_mean (24, 6, 6, 32, 16)
    poses = tf.squeeze(o_mean, axis=-3)

    # activation: (N, OH, OW, o) via squeeze o_activationis [24, 6, 6, 32]
    activations = tf.squeeze(o_activations, axis=[-3, -1])

  return poses, activations


def m_step(rr, votes, i_activations, beta_v, beta_a, inverse_temperature):
  """The M-Step in EM Routing from input capsules i to output capsule j.
  i: input capsules (32)
  o: output capsules (32)
  h: 4x4 = 16
  output spatial dimension: 6x6
  :param rr: routing assignments. shape = (kh x kw x i, o, 1) =(3x3x32, 32, 1) = (288, 32, 1)
  :param votes. shape = (N, OH, OW, kh x kw x i, o, 4x4) = (24, 6, 6, 288, 32, 16)
  :param i_activations: input capsule activation (at Level L). (N, OH, OW, kh x kw x i, 1, 1) = (24, 6, 6, 288, 1, 1)
     with dimensions expanded to match votes for broadcasting.
  :param beta_v: Trainable parameters in computing cost (1, 1, 1, 1, 32, 1)
  :param beta_a: Trainable parameters in computing next level activation (1, 1, 1, 1, 32, 1)
  :param inverse_temperature: lambda, increase over each iteration by the caller.

  :return: (o_mean, o_stdv, o_activation)
  """

  rr_prime = rr * i_activations

  # rr_prime_sum: sum over all input capsule i
  rr_prime_sum = tf.reduce_sum(rr_prime, axis=-3, keep_dims=True, name='rr_prime_sum')

  # o_mean: (24, 6, 6, 1, 32, 16)
  o_mean = tf.reduce_sum(
    rr_prime * votes, axis=-3, keep_dims=True
  ) / rr_prime_sum

  # o_stdv: (24, 6, 6, 1, 32, 16)
  o_stdv = tf.sqrt(
    tf.reduce_sum(
      rr_prime * tf.square(votes - o_mean), axis=-3, keep_dims=True
    ) / rr_prime_sum
  )

  # o_cost_h: (24, 6, 6, 1, 32, 16)
  o_cost_h = (beta_v + tf.log(o_stdv + epsilon)) * rr_prime_sum

  # o_cost: (24, 6, 6, 1, 32, 1)
  # o_activations_cost = (24, 6, 6, 1, 32, 1)
  # yg: This is done for numeric stability.
  # It is the relative variance between each channel determined which one should activate.
  o_cost = tf.reduce_sum(o_cost_h, axis=-1, keep_dims=True)
  o_cost_mean = tf.reduce_mean(o_cost, axis=-2, keep_dims=True)
  o_cost_stdv = tf.sqrt(
    tf.reduce_sum(
      tf.square(o_cost - o_cost_mean), axis=-2, keep_dims=True
    ) / o_cost.get_shape().as_list()[-2]
  )
  o_activations_cost = beta_a + (o_cost_mean - o_cost) / (o_cost_stdv + epsilon)

  # (24, 6, 6, 1, 32, 1)
  o_activations = tf.sigmoid(
    inverse_temperature * o_activations_cost
  )

  return o_mean, o_stdv, o_activations


def e_step(o_mean, o_stdv, o_activations, votes):
  """The E-Step in EM Routing.

  :param o_mean: (24, 6, 6, 1, 32, 16)
  :param o_stdv: (24, 6, 6, 1, 32, 16)
  :param o_activations: (24, 6, 6, 1, 32, 1)
  :param votes: (24, 6, 6, 288, 32, 16)

  :return: rr
  """

  o_p_unit0 = - tf.reduce_sum(
    tf.square(votes - o_mean) / (2 * tf.square(o_stdv)), axis=-1, keep_dims=True
  )

  o_p_unit2 = - tf.reduce_sum(
    tf.log(o_stdv + epsilon), axis=-1, keep_dims=True
  )

  # o_p is the probability density of the h-th component of the vote from i to j
  # (24, 6, 6, 1, 32, 16)
  o_p = o_p_unit0 + o_p_unit2

  # rr: (24, 6, 6, 288, 32, 1)cd

  zz = tf.log(o_activations + epsilon) + o_p
  rr = tf.nn.softmax(
    zz, dim=len(zz.get_shape().as_list())-2
  )

  return rr


def class_capsules(input, **params):
#(inputs, num_classes, iterations, batch_size, name):
    """
    :param inputs: ((24, 4, 4, 32, 4, 4), (24, 4, 4, 32))
    :param num_classes: 10
    :param iterations: 3
    :param batch_size: 24
    :param name:
    :return poses, activations: poses (24, 10, 4, 4), activation (24, 10).
    """
    num_classes = params['n_classes']
    iterations = 3

    inputs_poses, inputs_activations = input # (24, 4, 4, 32, 4, 4), (24, 4, 4, 32)
    print(inputs_poses.shape)

    batch_size = tf.shape(inputs_poses)[0]
    inputs_shape = inputs_poses.get_shape()
    spatial_size = int(inputs_shape[1])  # 4
    spatial_size_2 = int(inputs_shape[2])  # 4
    pose_size = int(inputs_shape[-1])    # 4
    i_size = int(inputs_shape[3])        # 32

    # inputs_poses (24*4*4=384, 32, 16)
    inputs_poses = tf.reshape(inputs_poses, shape=[batch_size*spatial_size*spatial_size_2, inputs_shape[-3], inputs_shape[-2]*inputs_shape[-2] ])

    with tf.variable_scope(params['name']) as scope:
        with tf.variable_scope('votes') as scope:
            # inputs_poses (384, 32, 16)
            # votes: (384, 32, 10, 16)
            votes = mat_transform(inputs_poses, num_classes, batch_size*spatial_size*spatial_size_2)
            # tf.logging.info(f"{name} votes shape: {votes.get_shape()}")

            # votes (24, 4, 4, 32, 10, 16)
            votes = tf.reshape(votes, shape=[batch_size, spatial_size, spatial_size_2, i_size, num_classes, pose_size*pose_size])

            # (24, 4, 4, 32, 10, 16)
            votes = coord_addition(votes, spatial_size, spatial_size_2)

            # tf.logging.info(f"{name} votes shape with coord addition: {votes.get_shape()}")

        with tf.variable_scope('routing') as scope:
            # beta_v and beta_a one for each output capsule: (1, 10)
            beta_v = tf.get_variable(
                name='beta_v', shape=[1, num_classes], dtype=tf.float32,
                initializer=initializers.xavier_initializer()
            )
            beta_a = tf.get_variable(
                name='beta_a', shape=[1, num_classes], dtype=tf.float32,
                initializer=initializers.xavier_initializer()
            )

            # votes (24, 4, 4, 32, 10, 16) -> (24, 512, 10, 16)
            votes_shape = votes.get_shape()
            votes = tf.reshape(votes, shape=[batch_size, votes_shape[1] * votes_shape[2] * votes_shape[3], votes_shape[4], votes_shape[5]] )

            # inputs_activations (24, 4, 4, 32) -> (24, 512)
            inputs_activations = tf.reshape(inputs_activations, shape=[batch_size,
                                                                       votes_shape[1] * votes_shape[2] * votes_shape[3]])

            # votes (24, 512, 10, 16), inputs_activations (24, 512)
            # poses (24, 10, 16), activation (24, 10)
            poses, activations = matrix_capsules_em_routing(
                votes, inputs_activations, beta_v, beta_a, iterations, name='em_routing'
            )

        # poses (24, 10, 16) -> (24, 10, 4, 4)
        poses = tf.reshape(
            poses, shape = [batch_size, num_classes, pose_size, pose_size],
            name = 'poses'
        )
        activations = tf.identity(activations, name = 'activations')

        # poses (24, 10, 4, 4), activation (24, 10)
        return poses, activations


def spread_loss(input, **params):
    poses, activations = input
    target = to_tensor(params['target'])
    margin = to_tensor(params['margin'])

    with tf.variable_scope(params['name']):
        poses = tf.identity(poses, name = 'poses')
        activations = tf.identity(activations, name = 'activations')
        target_one_hot = tf.one_hot(target, depth = activations.shape[-1])

        predicted_classes = tf.argmax(
            activations, axis = -1,
            name = 'predicted_classes'
        )
        correct_predictions = tf.cast(
            tf.equal(predicted_classes, target),
            tf.float32,
            name = 'correct_predictions'
        )
        accuracy_mean = tf.reduce_mean(
            correct_predictions,
            name = 'accuracy_mean'
        )
        accuracy_sum = tf.reduce_sum(
            correct_predictions,
            name = 'accuracy_sum'
        )

        loss = (
            tf.reduce_sum(activations * target_one_hot, axis = - 1, keep_dims = True) -
            activations
        )
        loss = loss * (1. - target_one_hot)
        loss = tf.reduce_sum(
            tf.square(tf.maximum(margin - loss, 0.)),
            name = 'output'
        )

    return loss


def coord_addition(votes, H, W):
    """Coordinate addition.

    :param votes: (24, 4, 4, 32, 10, 16)
    :param H, W: spaital height and width 4

    :return votes: (24, 4, 4, 32, 10, 16)
    """
    coordinate_offset_hh = tf.reshape(
      (tf.range(H, dtype=tf.float32) + 0.50) / H, [1, H, 1, 1, 1]
    )
    coordinate_offset_h0 = tf.constant(
      0.0, shape=[1, H, 1, 1, 1], dtype=tf.float32
    )
    coordinate_offset_h = tf.stack(
      [coordinate_offset_hh, coordinate_offset_h0] + [coordinate_offset_h0 for _ in range(14)], axis=-1
    )  # (1, 4, 1, 1, 1, 16)

    coordinate_offset_ww = tf.reshape(
      (tf.range(W, dtype=tf.float32) + 0.50) / W, [1, 1, W, 1, 1]
    )
    coordinate_offset_w0 = tf.constant(
      0.0, shape=[1, 1, W, 1, 1], dtype=tf.float32
    )
    coordinate_offset_w = tf.stack(
      [coordinate_offset_w0, coordinate_offset_ww] + [coordinate_offset_w0 for _ in range(14)], axis=-1
    ) # (1, 1, 4, 1, 1, 16)

    # (24, 4, 4, 32, 10, 16)
    votes = votes + coordinate_offset_h + coordinate_offset_w

    return votes

