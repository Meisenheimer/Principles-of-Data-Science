num_iter = 150

fp = open("run15.sh", "w")
for data_dir in ["node=15_size=4800_step=4800_rho=0.100000", "node=15_size=480_step=180_rho=0.100000", "node=15_size=4800_step=4800_rho=0.010000", "node=15_size=480_step=180_rho=0.010000"]:
    print(f"nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/{data_dir}/ --output ./nu_{data_dir}_num_iter={num_iter}.log --num_iter {num_iter} &", file=fp)
    print(f"nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/{data_dir}/ --output ./linear_{data_dir}_num_iter={num_iter}.log --num_iter {num_iter} --Linear LINEAR&", file=fp)

fp = open("run25.sh", "w")
for data_dir in ["node=25_size=4800_step=4800_rho=0.100000", "node=25_size=480_step=180_rho=0.100000", "node=25_size=4800_step=4800_rho=0.010000", "node=25_size=480_step=180_rho=0.010000"]:
    print(f"nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/{data_dir}/ --output ./nu_{data_dir}_num_iter={num_iter}.log --num_iter {num_iter} &", file=fp)
    print(f"nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/{data_dir}/ --output ./linear_{data_dir}_num_iter={num_iter}.log --num_iter {num_iter} --Linear LINEAR&", file=fp)

fp = open("run50.sh", "w")
for data_dir in ["node=50_size=4800_step=4800_rho=0.100000", "node=50_size=480_step=180_rho=0.100000", "node=50_size=4800_step=4800_rho=0.010000", "node=50_size=480_step=180_rho=0.010000"]:
    print(f"nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/{data_dir}/ --output ./nu_{data_dir}_num_iter={num_iter}.log --num_iter {num_iter} &", file=fp)
    print(f"nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/{data_dir}/ --output ./linear_{data_dir}_num_iter={num_iter}.log --num_iter {num_iter} --Linear LINEAR&", file=fp)
