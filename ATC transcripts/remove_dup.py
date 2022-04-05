
# outfile=open("transcripts_no_dup.txt","w")
# lines_seen=set()
# with open("testset.txt","r") as f:
#     for line in f:
#         line=line.lower()
#         if not line.isspace():
#             line=line.rstrip("\n")
#             if line not in lines_seen:
#                 lines_seen.add(line)
#                 outfile.write(line+"\n")    
#     outfile.close()


import re
import num2words


with open('callsign2.txt') as f_input:
    text = f_input.read()


text = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), text)

with open('callsign2_words.txt', 'w') as f_output:
    f_output.write(text)