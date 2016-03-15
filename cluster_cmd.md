/usr/local/spark-1.6.0-bin-hadoop1/ec2/spark-ec2 -k bigdata -i /home/han/.api_key/bigdata.pem -r us-east-1 instance-type=m3.large -m m3.large -s 6 launch han_project

ssh -i ~/.api_key/tweet.pem ec2-52-90-48-34.compute-1.amazonaws.com



scp -i ~/.api_key/bigdata.pem install_scripts/install_these root@ec2-52-90-48-34.compute-1.amazonaws.com:/root/.


/usr/local/spark-1.6.0-bin-hadoop1/ec2/spark-ec2 -k bigdata -i /home/han/.api_key/bigdata.pem -r us-east-1 -s 6 login han_project



aws s3 cp 07 s3://han.tweets.bucket/tweets --recursive



[path to ec2] -k testkey -i /Users/miahunsicker/Desktop/testkey.pem -r us-west-1 instance-type=m3.large -m m3.large -s 12 launch tweettest

/usr/local/spark-1.6.0-bin-hadoop1/ec2/spark-ec2 -k bigdata -i /home/han/.api_key/bigdata.pem -r us-east-1 stop han_project
/usr/local/spark-1.6.0-bin-hadoop1/ec2/spark-ec2 -k bigdata -i /home/han/.api_key/bigdata.pem -r us-east-1 start han_project



IPYTHON_OPTS="notebook --ip=0.0.0.0" /root/spark/bin/pyspark --executor-memory 4G --driver-memory 4G &








# constructing w2v

w2v = dict(WP.w2v.getVector())

for key, val in w2v.iteritems():
    w2v[key] = list[val]

w2v_idx = np.array(w2v.keys())
w2v_vect = np.array(w2v.values())
