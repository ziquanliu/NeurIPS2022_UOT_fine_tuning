import numpy as np
import pickle
import os
from sinkhorn_unbalanced import sinkhorn_knopp_unbalanced
import random


def sample_OT_Similarity(select_classes,p_m,imagenet_path,dataset,which_pt):
    # K_t*dim
    target = pickle.load(open('./class_mean_'+which_pt+'/'+dataset+'/class_mean_unnorm.pkl','rb'))



    K_t = target.shape[0]
    # K_s*dim
    source = pickle.load(open('./class_mean_'+which_pt+'/ImageNet/class_mean_unnorm.pkl','rb'))

    K_s = source.shape[0]

    target_dist = np.ones(K_t)
    # target_dist = target_dist/np.sum(target_dist)

    source_dist = np.ones(K_s)
    # source_dist = source_dist/np.sum(source_dist)


    # normalize features, this seems to matter a lot
    



    # L2 distance
#     p_m = 0.1
    #target = target*(1./np.linalg.norm(target,axis=-1,keepdims=True))
    #source = source*(1./np.linalg.norm(source,axis=-1,keepdims=True))
    #M=np.sqrt(np.sum(np.square(np.expand_dims(target,axis=0)-np.expand_dims(source,axis=1)),axis = -1))/p_m

    # cos distance
    # general set, supervised
    # general set, unsupervised
    
    target = target*(1./np.linalg.norm(target,axis=-1,keepdims=True))
    source = source*(1./np.linalg.norm(source,axis=-1,keepdims=True))
    M = (-np.sum(np.multiply(np.expand_dims(target,axis=0),np.expand_dims(source,axis=1)),axis = -1)+1.0)/p_m

    # K_s*K_t
    #print(M)
    #print(target_dist.shape)
    #print(source_dist.shape)
    epsilon_ent = 1.0
    T = sinkhorn_knopp_unbalanced(source_dist, target_dist, M, epsilon_ent, 1.0, 100.0 )

    # print(np.matmul(T,np.ones((K_t,1))).transpose())
    # print(np.matmul(T.transpose(),np.ones((K_s,1))).transpose())


    source_dist_importance = np.squeeze(np.matmul(T,np.ones((K_t,1))))
    #print(source_dist_importance)
    #print(np.sort(source_dist_importance))
    #print(np.amax(source_dist_importance))
    #print(np.amin(source_dist_importance))

    high_prob_ind = np.argsort(source_dist_importance)[-select_classes:]
    #print(np.sort(high_prob_ind))

    class_concept = pickle.load(open('class_concept.pkl','rb'))
    class_name_all = pickle.load(open('class_names.pkl','rb'))
    class_path_list = pickle.load(open('class_path.pkl','rb'))
    #print(len(class_path_list))
    #print(len(class_concept))
    count_bird = 0
    select_class_concept = []
    select_class_name = []
    for class_id in high_prob_ind:
    #     print(class_concept[class_id])
        select_class_concept.append(class_concept[class_id])
        select_class_name.append(class_name_all[class_id])
        
        if class_concept[class_id] == 'bird':
            count_bird += 1
    #print(count_bird)
    #print(len(select_class_concept))
    images_bird = []
    for class_ind in high_prob_ind:
        # print(class_ind)
        class_path = os.path.join(imagenet_path, class_path_list[class_ind])
        images_bird = images_bird + os.listdir(class_path)

    random.shuffle(images_bird)
    with open('supervised_selection/'+dataset+'/OT_unnorm_cos_imagenet_OT_select_' + str(
            select_classes) + '_classes_train_samples.txt', 'a') as write_file:
        for filename in images_bird:
            write_file.write(filename + '\n')

    for class_name in zip(select_class_concept,select_class_name):
        print(class_name[0]+', '+class_name[1])
    print(count_bird)
    return count_bird,np.sort(source_dist_importance)




imagenet_path = 'imagenet-1k/train' # path of imagenet
num_selected_classes = 100 # number of selected classes from imagenet
p_m_list=[0.1] # hyperparameter in distance function
count_list= [] # if CUB, count the number of selected bird classes
rank_vec_list = [] # P1 vector
for p_m in p_m_list:
    bird_count,rank_vec = sample_OT_Similarity(num_selected_classes,p_m,imagenet_path,'CUB','supervised')
    count_list.append(bird_count)
    rank_vec_list.append(rank_vec)
