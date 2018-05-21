import itchat
import pickle
itchat.login()
friends = itchat.get_friends(update=True)[0:]

with open('friends_info.pkl','wb') as file:
	pickle.dump(friends, file)
