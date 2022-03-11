import random 
filename='transcripts_no_dup.txt'
outfile=open("testset.txt","w")
def choose_line(filename): 
#"""Choose a line at random from the text file""" 
    with open(filename, 'r') as file:
            lines = file.readlines() 
            random_line = random.choice(lines)
            outfile.write(random_line)

for i in range(1,7001):
    choose_line(filename)