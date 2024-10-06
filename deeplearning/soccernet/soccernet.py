from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/home/wolf/datasets/soccernetv3")
mySoccerNetDownloader.downloadDataTask(task="reid", split=["train", "valid", "test", "challenge"])
