python_exe=python3.9

# Experiments for impact of threshold
# $python_exe experiments.py --dataset flickr --nbr_exp 10 --attack dice fgsm nettack --proportion 0.5 --thresholds 0 100 10 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.1 --expe_name impact_of_threshold



# #Experiments for impact of proportion
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack dice fgsm nettack --proportion 0.0 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.1 --expe_name impact_of_proportion
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.1 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.1 --expe_name impact_of_proportion
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.2 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.1 --expe_name impact_of_proportion
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.3 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.1 --expe_name impact_of_proportion
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.4 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.1 --expe_name impact_of_proportion
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.5 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.1 --expe_name impact_of_proportion
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.6 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.1 --expe_name impact_of_proportion
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.7 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.1 --expe_name impact_of_proportion
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.8 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.1 --expe_name impact_of_proportion
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.9 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.1 --expe_name impact_of_proportion
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 1.0 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.1 --expe_name impact_of_proportion


# # # #Experiments for impact of perturbation rate
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.5 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.0 --expe_name impact_of_perturbation_rate
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.5 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.1 --expe_name impact_of_perturbation_rate
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.5 --thresholds 0 70 1 --ptb_rate_graph1 0.0 --ptb_rate_graph2 0.2 --expe_name impact_of_perturbation_rate
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.5 --thresholds 0 70 1 --ptb_rate_graph1 0.1 --ptb_rate_graph2 0.1 --expe_name impact_of_perturbation_rate
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.5 --thresholds 0 70 1 --ptb_rate_graph1 0.1 --ptb_rate_graph2 0.2 --expe_name impact_of_perturbation_rate
# python3.9 experiments.py --dataset flickr --nbr_exp 10 --attack nettack fgsm dice --proportion 0.5 --thresholds 0 70 1 --ptb_rate_graph1 0.2 --ptb_rate_graph2 0.2 --expe_name impact_of_perturbation_rate

