import subprocess
import sys
import os

if __name__=="__main__":
	cmd = "rsync -avL --progress -e \"ssh -i /home/ubuntu/aws_ds8key.pem\" ubuntu@ec2-54-162-31-22.compute-1.amazonaws.com:~/project/Rank_a_Golfer/ranks/ /home/ubuntu/project/Rank_a_Golfer/ranks/"
	os.system(cmd)
	cmd = "rsync -avL --progress -e \"ssh -i /home/ubuntu/aws_ds8key.pem\" ubuntu@ec2-54-162-31-22.compute-1.amazonaws.com:~/project/Rank_a_Golfer/cats/ /home/ubuntu/project/Rank_a_Golfer/cats/"
	os.system(cmd)
	subprocess.call(["python","crawler.py"])
