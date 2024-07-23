source ~/anaconda3/etc/profile.d/conda.sh

attack_type='inverse'  # forward, inverse
# tag='random_attack'

for tag in 1_obj_1 1_obj_2 1_obj_4 1_obj_5
do
    for query_num in 100
    do
        dataset_root=/home/kyrie/Desktop/attack_pred

        # ### det attack ###
        # conda activate pixor_nuscs
        # cd /home/kyrie/Desktop/WorkSpace/lidar/PIXOR_nuscs/srcs
        # python -W ignore attack.py \
        #     --attack_type ${attack_type} --query_num ${query_num} --tag ${tag}
        # ### det attack ###


        # ### pred attack ###
        # if [ $attack_type == 'forward' ]
        # then

        #     # pred attack
        #     conda activate trajectron++
        #     cd /home/kyrie/Desktop/WorkSpace/prediction/Trajectron-plus-plus/experiments/nuScenes
            
        #     python -W ignore forward.py \
        #         --dataset_root ${dataset_root} \
        #         --attack_type ${attack_type} --query_num ${query_num} --tag ${tag}

        # elif [ $attack_type == 'inverse' ]
        # then
        #     conda activate trajectron++
        #     cd /home/kyrie/Desktop/WorkSpace/prediction/Trajectron-plus-plus/experiments/nuScenes
            
        #     # pred attack
        #     python inverse.py \
        #         --dataset_root ${dataset_root} \
        #         --attack_type ${attack_type} --query_num ${query_num} --tag ${tag}
        #     # python inverse_diffplan.py \
        #     #     --attack_type ${attack_type} --query_num ${query_num} --tag ${tag}

        #     # candidate set matching
        #     python matching.py \
        #         --dataset_root ${dataset_root} \
        #         --attack_type ${attack_type} --query_num ${query_num} --tag ${tag}

        #     # second stage refinement
        #     conda activate pixor_nuscs
        #     cd /home/kyrie/Desktop/WorkSpace/lidar/PIXOR_nuscs/srcs
        #     python -W ignore refine_iterative.py \
        #         --attack_type ${attack_type} --query_num ${query_num} --tag ${tag}
        #     # python -W ignore refine.py \
        #     #     --attack_type ${attack_type} --query_num ${query_num} --tag ${tag}
        # fi
        # ### pred attack ###


        ### det eval ###
        conda activate pixor_nuscs
        cd /home/kyrie/Desktop/WorkSpace/lidar/PIXOR_nuscs/srcs
        python -W ignore eval.py \
            --attack_type ${attack_type} --query_num ${query_num} --tag ${tag}
        ### det eval ###


        ### pred eval ###
        conda activate trajectron++
        cd /home/kyrie/Desktop/WorkSpace/prediction/Trajectron-plus-plus/experiments/nuScenes
        python eval_multivelo.py \
            --dataset_root ${dataset_root} \
            --attack_type ${attack_type} --query_num ${query_num} --tag ${tag} \
            --track

        # # similar with eval_multivelo, but consider different planning result under different velocitites
        # python eval_test_diffPlan.py \
        #     --attack_type ${attack_type} --query_num ${query_num} --tag ${tag}
        
        # # find the best candidate for each velocity
        # python eval_test_veloSpecific.py \
        #     --attack_type ${attack_type} --query_num ${query_num} --tag ${tag}
        ## pred eval ###

        # # ### percep-guard ###
        # # conda activate pixor_nuscs
        # # cd /home/kyrie/Desktop/WorkSpace/lidar/PIXOR_nuscs/srcs
        # # python -W ignore eval_export_2d.py \
        # #     --attack_type ${attack_type} --query_num ${query_num} --tag ${tag}
        # # ### percep-guard ###
    done
done


# python eval_multivelo.py --attack_type inverse --query_num 500 --tag 1 --plot

# python eval_defense.py --attack_type inverse --query_num 100 --tag 1