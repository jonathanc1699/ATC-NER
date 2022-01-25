arr = ['zero','one','two','three','four','five','six','seven','eight','niner']
 
def number_2_word(n):
 
    # If all the digits are encountered return blank string
    if(n==0):
        return ""
     
    else:
        # compute spelling for the last digit
        small_ans = arr[n%10]
 
        # keep computing for the previous digits and add the spelling for the last digit
        ans = number_2_word(int(n/10)) + small_ans + " "
     
    # Return the final answer
    return ans

outfile = open("cs_cleaned.txt","w")
lines_seen = set() # holds lines already seen

with open("cs.txt","r") as f:
    for line in f:
        line=line.rstrip("\n")
        if line not in lines_seen:
            lines_seen.add(line)
            x =int(line.split(" ")[-1])
            n=number_2_word(x)
            y=line.replace(line.split(" ")[-1],n)
            outfile.write(y+"\n")    
    outfile.close()