import argparse
import os

import crypten

parser = argparse.ArgumentParser(description="Crypten TTP Demo")

parser.add_argument(
    "-a", "--host_addr", type=str, default="127.0.0.1", help="The host address to bind."
)
parser.add_argument("-p", "--port", type=int, default="29500", help="The port to bind.")
parser.add_argument("-r", "--rank", type=int, default="0", help="Rank of the server.")

args = vars(parser.parse_args())
os.environ["RENDEZVOUS"] = "env://"
os.environ["WORLD_SIZE"] = str(2)
os.environ["RANK"] = str(args["rank"])
os.environ["MASTER_ADDR"] = args["host_addr"]
os.environ["MASTER_PORT"] = str(args["port"])

crypten.mpc.provider.ttp_provider.TTPServer()
