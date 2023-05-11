
outfile=open("callsigns_2_try.txt","w")
with outfile as f:
    for i in range(10,100):
        f.write(str(i) + "\n")
        for j in range(10,100):
            f.write(str(i)+" " +str(j)+'\n')
    outfile.close()