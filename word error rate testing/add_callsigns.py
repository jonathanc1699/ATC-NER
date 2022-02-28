## Name: Jonathan Chang
## Role: Intern for ATC LM adaptation
## Date: 20/02/2022
## Description: This script is used to add callsigns to the current hotwords list for testing. Ensure that both files are in the same directory as this program.
## Last change: inital creation


#declaring callsigns file and hotwords file
firstfile = input("Enter the name of hotwords list txt file: ")
secondfile = input("Enter the name of callsigns txt file: ")
# opening both files in read only mode to read initial contents
f1 = open(firstfile, 'r')
f2 = open(secondfile, 'r')
 
# printing the contents of the file before appending
print('content of first file before appending : \n', f1.read())
print('content of second file before appending : \n', f2.read())
 
# closing the files
f1.close()
f2.close()
 
# opening first file in append mode and second file in read mode
f1 = open(firstfile, 'a+')
f2 = open(secondfile, 'r')
 
# appending the contents of the second file to the first file
f1.write(f2.read())
 
# relocating the cursor of the files at the beginning
f1.seek(0)
f2.seek(0)
 
# printing the contents of the files after appendng
print('content of first file after appending : \n', f1.read())
print('content of second file after appending : \n', f2.read())
 
# closing the files
f1.close()
f2.close()