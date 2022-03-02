import random 

d = {}
i=0
with open("Callsigns.txt") as f:
    for line in f:
        d[i] = line.rstrip("\n")
        i+=1
#print(d)

with open("cs.txt",'w') as f1:
    for j in range (100):
        for callsign in d:
            f1.write(d[callsign] + " " + str(random.randint(100,9999))+"\n")
        


    
