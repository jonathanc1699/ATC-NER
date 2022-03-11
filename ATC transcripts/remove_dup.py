
outfile=open("transcripts_no_dup.txt","w")
lines_seen=set()
with open("testset.txt","r") as f:
    for line in f:
        line=line.lower()
        if not line.isspace():
            line=line.rstrip("\n")
            if line not in lines_seen:
                lines_seen.add(line)
                outfile.write(line+"\n")    
    outfile.close()