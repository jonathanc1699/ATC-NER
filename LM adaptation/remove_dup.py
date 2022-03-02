outfile=open("text no dup.txt","w")
lines_seen=set()
with open("text copy.txt","r") as f:
    for line in f:
        line=line.rstrip("\n")
        if line not in lines_seen:
            lines_seen.add(line)
            outfile.write(line+"\n")    
    outfile.close()