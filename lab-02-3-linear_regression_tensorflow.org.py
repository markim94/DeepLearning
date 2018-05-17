# From https://www.tensorflow.org/get_started/get_started
import tensorflow as tf # 텐서플로우 라이브러리 사용

# Model parameters
W = tf.Variable([.3], tf.float32)  # W 변수에 32비트 실수형 변수 0.3
b = tf.Variable([-.3], tf.float32) # b 변수에 32비트 실수형 변수 -0.3
# Variable 내장함수를 사용하면 텐서플로 자체적으로 학습과정에서 값을 변경

# Model input and output
x = tf.placeholder(tf.float32) # x 변수에 32비트 실수형의 placeholder 지정
y = tf.placeholder(tf.float32) # y 변수에 32비트 실수형의 placeholder 지정
# placeholoer 내장함수는 만들어진 모델에 있어 값을 후에 던져줄 수 있음

linear_model = x * W + b # linear_model

# cost/loss function
loss = tf.reduce_sum(tf.square(linear_model - y))  # square은 제곱을 표현, reduce_sum은 평균치를 구하는 내장함수등을 통해 손실함수 표현

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01) # GradientDescentOptimizer 내장함수를 이용해 learning rate를 0.01로 지정
# learning rate를 크게 하면 타겟을 못잡을 수 있고 작으면 시간이 오래 걸리므로 적절한 값 지정

train = optimizer.minimize(loss) # optimizer가 loss를 learning rate(0.01) 간격마다 최소화시킴, 그 과정에서 variable로 지정된 W, b의 값 조정 발생

# training data
x_train = [1, 2, 3, 4] # 학습 x 데이터
y_train = [0, -1, -2, -3] # 학습 y 데이터

# training loop
init = tf.global_variables_initializer() # variable 실행 전 initializer를 해주어야 함
sess = tf.Session() # 그래프 실행하기 위한 세션
sess.run(init)  # reset values to wrong
for i in range(1000): # 1000번 반복
    sess.run(train, {x: x_train, y: y_train})   # placeholder로 지정했던 x, y에 feed_dict로 학습 데이터를 넣어 학습

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train}) # 1000번 학습 이후 조정된 W, b 값, 최종 오차 loss 세션
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss)) # 출력
