nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/node=25_size=4800_step=4800_rho=0.100000/ --output ./nu_node=25_size=4800_step=4800_rho=0.100000_num_iter=150.log --num_iter 150 &
nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/node=25_size=4800_step=4800_rho=0.100000/ --output ./linear_node=25_size=4800_step=4800_rho=0.100000_num_iter=150.log --num_iter 150 --Linear LINEAR&
nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/node=25_size=480_step=180_rho=0.100000/ --output ./nu_node=25_size=480_step=180_rho=0.100000_num_iter=150.log --num_iter 150 &
nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/node=25_size=480_step=180_rho=0.100000/ --output ./linear_node=25_size=480_step=180_rho=0.100000_num_iter=150.log --num_iter 150 --Linear LINEAR&
