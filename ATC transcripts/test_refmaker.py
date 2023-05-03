## Name: Jonathan Chang
## Role: Intern for ATC LM adaptation
## Date: 28/02/2022
## Description: This script is used to add callsigns to the current hotwords list for testing. Ensure that both files are in the same directory as this program.
## Last change: inital creation


#input names of callsigns file and hotwords file
secondfile = input("Enter the name of pre-processed txt file: ")

# # opening both files in read only mode to read initial contents
# f1 = open("testset.txt", 'r')
# f2 = open(secondfile, 'r')
 
# # printing the contents of the file before appending
# # print('content of first file before appending : \n', f1.readlines())
# # print('content of second file before appending : \n', f2.readlines())
 
# # closing the files
# f1.close()
# f2.close()

# opening first file in write mode and second file in read mode
f1 = open("s216-438.txt", 'w')
f2 = open(secondfile, 'r')
i=216
# find lines that start with "(TEXT " and them to test set
for line in f2:
    print(line)
    f1.write('atc-chn0-spk1-s'+ str(i) + " "+ line)
    i=i+1
 
# relocating the cursor of the files at the beginning
f1.seek(0)
f2.seek(0)
 
# printing the contents of the files after appendng
# print('content of first file after appending : \n', f1.read())
# print('content of second file after appending : \n', f2.read())
 
# closing the files
f1.close()
f2.close()