def count_combinations(pieces,target):
    result=list()
    def find_combination(start,target,c_omb):
        if target==0:
           result.append(tuple(c_omb))
           return
        if target<0:
            return
        for i in range(start,len(pieces)):
            c_omb.append(pieces[i])
            find_combination(i+1,target-pieces[i],c_omb)
            print(c_omb)
            c_omb.pop()
    pieces.sort()
    find_combination(0,target,[])
    return result
pieces=[10,1,2,7,6,1,5]
target=8
combination=count_combinations(pieces,target)
print(len(set(combination)))
print(combination)