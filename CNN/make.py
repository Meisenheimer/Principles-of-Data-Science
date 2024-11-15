epochs = 64
num_iter = 150

fp = open("run15.sh", "w")
for data_dir in ["node=15_size=4800_step=4800_rho=0.100000", "node=15_size=480_step=180_rho=0.100000"]:
    for channel in [1, 2, 4]:
        for lr in [1e-3]:
            for dropout in [0.0, 0.1]:
                print(f"nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/{data_dir}/ --output_dir ./output/CNN_{data_dir}_dropout={dropout}_epochs={epochs}_num_iter={num_iter}_lr={lr}_channel={channel}/ --dropout {dropout} --epochs {epochs} --num_iter {num_iter} --lr {lr} --channel {channel} &", file=fp)
                if (data_dir == "node=15_size=480_step=180_rho=0.100000"):
                    print(f"nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/{data_dir}/ --output_dir ./output/CNN_{data_dir}_dropout={dropout}_epochs={epochs}_num_iter={num_iter}_lr={lr}_choose_channel={channel}/ --dropout {dropout} --epochs {epochs} --num_iter {num_iter} --lr {lr} --channel {channel} --choose CHOOSE &", file=fp)

fp = open("run25.sh", "w")
for data_dir in ["node=25_size=4800_step=4800_rho=0.100000", "node=25_size=480_step=180_rho=0.100000"]:
    for channel in [1, 2, 4]:
        for lr in [1e-3]:
            for dropout in [0.0, 0.1]:
                print(f"nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/{data_dir}/ --output_dir ./output/CNN_{data_dir}_dropout={dropout}_epochs={epochs}_num_iter={num_iter}_lr={lr}_channel={channel}/ --dropout {dropout} --epochs {epochs} --num_iter {num_iter} --lr {lr} --channel {channel} &", file=fp)
                if (data_dir == "node=25_size=480_step=180_rho=0.100000"):
                    print(f"nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/{data_dir}/ --output_dir ./output/CNN_{data_dir}_dropout={dropout}_epochs={epochs}_num_iter={num_iter}_lr={lr}_choose_channel={channel}/ --dropout {dropout} --epochs {epochs} --num_iter {num_iter} --lr {lr} --channel {channel} --choose CHOOSE &", file=fp)

fp = open("run50.sh", "w")
for data_dir in ["node=50_size=4800_step=4800_rho=0.100000", "node=50_size=480_step=180_rho=0.100000"]:
    for channel in [1, 2, 4]:
        for lr in [1e-3]:
            for dropout in [0.0, 0.1]:
                print(f"nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/{data_dir}/ --output_dir ./output/CNN_{data_dir}_dropout={dropout}_epochs={epochs}_num_iter={num_iter}_lr={lr}_channel={channel}/ --dropout {dropout} --epochs {epochs} --num_iter {num_iter} --lr {lr} --channel {channel} &", file=fp)
                if (data_dir == "node=50_size=480_step=180_rho=0.100000"):
                    print(f"nohup python train.py --data_file_dir ../Data/ --data_dir ../Data/functional_connectivity/{data_dir}/ --output_dir ./output/CNN_{data_dir}_dropout={dropout}_epochs={epochs}_num_iter={num_iter}_lr={lr}_choose_channel={channel}/ --dropout {dropout} --epochs {epochs} --num_iter {num_iter} --lr {lr} --channel {channel} --choose CHOOSE &", file=fp)
