######################################################
#  prunded gradient descent for MNIST linear network
# BN Kausik May 22 2024
######################################################




import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

np.set_printoptions(precision=4);
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train=np.float16(x_train);
y_train=np.float16(y_train);
x_test=np.float16(x_test);
y_test=np.float16(y_test);

N_units=128
n_epochs=10
v_flag=0
occam_flag=1;
p_flag=0;
lr=0.4

opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
for i in range (0,len(opts)):
    if opts[i]=="-p": p_flag=int(args[i]);
    if opts[i]=="-v": v_flag=int(args[i]);
    if opts[i]=="-o": occam_flag=int(args[i]);
    if opts[i]=="-n_epochs": n_epochs=int(args[i]);
    if opts[i]=="-N_units": N_units=int(args[i]);
    if opts[i]=="-lr": lr=float(args[i]);

print("N_units",N_units,"n_epochs",n_epochs);
print("v_flag",v_flag,"occam_flag",occam_flag,"p_flag",p_flag);
print("lr",lr)

Q_1=tf.Variable(np.ones((784,N_units)))  # quantization matrix
Q_3=tf.Variable(np.ones((N_units,10)))  # quantization matrix
Q_1_old=np.ones((784,N_units))  # old quantization matrix
Q_3_old=np.ones((N_units,10))  # old quantization matrix

class quant_1(tf.keras.constraints.Constraint):
    def __call__(self, w):
        if occam_flag: return tf.where(Q_1 ==0.0,0.0,w)
        else: return w

class quant_3(tf.keras.constraints.Constraint):
    def __call__(self, w):
        if occam_flag: return tf.where(Q_3 ==0.0,0.0,w)
        else: return w


model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),
	tf.keras.layers.Dense(N_units, activation='relu', kernel_constraint=quant_1()),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(10, kernel_constraint=quant_3())
])

print(model.summary());


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

lr_0=lr
err_log=np.zeros((2,n_epochs,2))
wts=np.zeros(n_epochs,dtype=int)
max_N=N_units+10
max_N+=model.layers[1].get_weights()[0].size
max_N+=model.layers[3].get_weights()[0].size
i=0
for i in range(n_epochs):
    model.fit(x_train, y_train, epochs=1,verbose=v_flag)
    err_log[0,i]=model.evaluate(x_train,  y_train, verbose=v_flag)
    err_log[1,i]=model.evaluate(x_test,  y_test, verbose=v_flag)
    if occam_flag:
        wts[i]=N_units+10
        wts[i]+=int(tf.math.reduce_sum(Q_1).numpy())
        wts[i]+=int(tf.math.reduce_sum(Q_3).numpy())
        if i>1:
            zeta=(err_log[0,i,0]-err_log[0,i-1,0])/(err_log[0,i-1,0]-err_log[0,i-2,0])
            print("zeta",zeta)
            lr=np.clip(zeta*lr,0.1*lr_0,lr_0)
        W_t=np.abs(model.layers[1].get_weights()[0])
        q= np.quantile(W_t[W_t!=0],lr)
        W_t=np.where(W_t<q,0,1)
        Q_1.assign(W_t)

        W_t=np.abs(model.layers[3].get_weights()[0])
        q= np.quantile(W_t[W_t!=0],lr)
        W_t=np.where(W_t<q,0,1)
        Q_3.assign(W_t)
    else: wts[i]=max_N
    print("#### ",i,wts[i],max_N,err_log[0,i,0],err_log[1,i,0],err_log[0,i,1],err_log[1,i,1],lr)

opt_iter=err_log[0,:,0].argmin()
opt_N=wts[opt_iter]
print("opt_iter",opt_iter,"opt_N",opt_N)


if p_flag:
    plt.figure(1)
    plt.plot(err_log[0,:,0], label='train_loss',marker="o")
    plt.plot(err_log[1,:,0], label='test_loss',marker="s")
    plt.text(0.03, 0.85, "loss: " +str(N_units)+" units,"+"{:.1f}".format(100*opt_N/max_N)+"% opt wts" ,transform=plt.gca().transAxes,  bbox=dict(facecolor='white', alpha=0.5))
    plt.figure(2)
    plt.plot(err_log[0,:,1], label='train_acc',marker="o")
    plt.plot(err_log[1,:,1], label='test_acc',marker="s")
    plt.text(0.03, 0.85, "accuracy: " +str(N_units)+" units,"+"{:.1f}".format(100*opt_N/max_N)+"% opt wts" ,transform=plt.gca().transAxes,  bbox=dict(facecolor='white', alpha=0.5))
    plt.show();
