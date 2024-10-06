from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/home/wolf/datasets/SoccerNet")
# mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
