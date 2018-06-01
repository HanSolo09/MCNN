import tensorflow as tf
import cv2
import tifffile
import input_data
import numpy as np
import model
import glob
import datetime

R_size = (15,15)
N_CLASSES = 7
BATCH_SIZE = 1

def evaluate_rs_images():
    starttime = datetime.datetime.now()
    image_dir = "D:\\Thesis\\Pathches\\6110_4_0\\*.tif"


    patches_dir = glob.glob(image_dir)
    #gridcode_list = list(map(lambda  filename:((filename.split("\\")[6]).split(".")[0]),patches_dir))

    #gridcode_list = list(map(int,gridcode_list)) # string to int
    #patches_matrix_list = []
    jishu = 0
    #jishu2 =0
    #for i in patches_dir:
        #patch_matrix = tifffile.imread(i)
        #patch_matrix = np.array(patch_matrix, dtype= np.uint16)
        #patch_matrix = cv2.resize(patch_matrix,R_size, interpolation=cv2.INTER_CUBIC)
        #patches_matrix_list.append(patch_matrix)
        #jishu = jishu +1
        #print('reading the %d'%(jishu))
    #print('zhuanhuanzhong...')
    #patches_matrix_list = np.array(patches_matrix_list)
    #gridcode_list = tf.cast(gridcode_list, tf.int32)
    #patches_matrix_list = tf.cast(patches_matrix_list,tf.int32)
    #print('jianli dui lie... ')
    #queue = tf.train.slice_input_producer([patches_matrix_list,gridcode_list], num_epochs = 1, shuffle=False )
    #patches = queue[0]
    #gridcode = queue[1]
    #patches = tf.image.per_image_standardization(patches)
    #print('jianli batch...')
    #patches_batch= tf.train.batch(patches, batch_size = BATCH_SIZE,
                                                   #num_threads=2, capacity = BATCH_SIZE*50)
    #patches_batch = tf.cast(patches_batch,tf.float32)
    #print('huoqu batch wancheng')

    Class_list =[]

    x1 = tf.placeholder(tf.int32,shape = [15,15,11],name = "the_patch")
    x2 = tf.image.per_image_standardization(x1)
    x3 = tf.reshape(x2,[1, 15, 15, 11])
    print(x3)


    x4 =tf.cast(x3, tf.float32)


    logit = model.inference(x4, BATCH_SIZE, N_CLASSES)
    print(logit)

    logit = tf.nn.softmax(logit)
    Class = tf.argmax(logit, 1)
    logs_train_dir = 'D:\\Thesis\\classify\\6110_4_0\\logs\\train'

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init_local_op = tf.initialize_local_variables()


    print( 'qidong session:')
    with tf.Session(config=config) as sess:
        sess.run(init_local_op)
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in patches_dir:
            patch_matrix = tifffile.imread(i)
            patch_matrix = np.array(patch_matrix, dtype=np.uint16)
            patch_matrix = cv2.resize(patch_matrix, R_size, interpolation=cv2.INTER_CUBIC)
            #patch_matrix = patch_matrix.reshape(1,25,25,4)


            Patches_Class = sess.run(Class,feed_dict={x1:patch_matrix})
            Class_list.append(Patches_Class)
            print(Patches_Class)






        endtime = datetime.datetime.now()
        print((endtime - starttime).seconds)
        print(len(Class_list))
        Class_list = np.array(Class_list)
        np.savetxt('Class_list',Class_list)
