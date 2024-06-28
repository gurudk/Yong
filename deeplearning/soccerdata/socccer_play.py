import soccerdata as sd

fbref = sd.FBref("INT-European Championship", 2024)
# mr = fbref.read_player_match_stats(match_id="bd775264")
# print(mr)

mr = fbref.read_team_match_stats(team="Germany")
print(mr)
sc = fbref.read_schedule()
print(sc.head())
