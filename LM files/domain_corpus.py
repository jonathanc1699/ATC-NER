non_cs=1363
num_lines=sum(1 for line in open('HotWords.txt'))
# open both files
with open('HotWords.txt','r') as firstfile, open('test.txt','a') as secondfile:  
    # read content from first file
    for i in range(1,non_cs):
        line=firstfile.readline()
        line=line.rstrip()
        # append content to second file
        secondfile.write(line + " " + "1000\n")
    
    for i in range(1363,num_lines):
        line=firstfile.readline()
        line=line.rstrip()
        #append content to second file
        secondfile.write(line + " " + "50\n")
    