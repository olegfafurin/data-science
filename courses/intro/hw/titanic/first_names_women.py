abbrs = ["Miss.", "Mrs.", "Mlle."]

def get(name):
    words = name.split()
    for abbr in abbrs:
        if abbr in words:
            return(words[words.index(abbr) + 1])
            break;
    begin = name.find("(")
    return(name[begin + 1 : name.find(" ", begin)])


names = ["Heikkinen, Miss. Lain", "Nicola-Yarred, Miss. Jamil", "O'Dwyer, Miss. Ellen \"Nellie\""]
for name in names:
    print(get(name))