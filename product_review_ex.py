def questionnaire():
  userInput=[]

  name = input(f"Hello! What's your name? ")
  print(f"Hi {name}! I will ask you a series 5 of questions regarding recent product X purchase.")
  print("--------------------------------------------------------------\n")

  userInput1 = input("1. How would you describe your favorite feature(s) of the product? ")
  userInput.append(userInput1)
  print(" ")

  userInput2 = input("2. Why did you choose our product over the competitors'? ")
  userInput.append(userInput2)
  print(" ")

  userInput3 = input("3. What would you want to see improved in an upgraded version?  ")
  userInput.append(userInput3)
  print(" ")

  userInput4 = input("4.How likely is it that you would recommend this product to others? ")
  userInput.append(userInput4)
  print(" ")

  userInput5 = input(f"5. {name}, is there anything else you would add before we end the session? ")
  userInput.append(userInput5)
  print("-------------------------------------------------------------------------- ")

  response = " ".join(userInput)
  return response
