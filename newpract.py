def count_combinations(pieces, target):
    result = set()
    pieces.sort()
    
    # Stack to keep track of the state: (start index, current combination, current sum)
    stack = [(0, [], 0)]
    
    while stack:
        start, c_omb, current_sum = stack.pop()
        
        # If the current sum equals the target, add the combination to the result set
        if current_sum == target:
            result.add(tuple(c_omb))
            continue
        
        # Iterate through remaining pieces starting from 'start'
        for i in range(start, len(pieces)):
            # Skip adding if the sum would exceed the target
            if current_sum + pieces[i] > target:
                continue
            
            # Skip duplicates by checking the previous element
            if i > start and pieces[i] == pieces[i - 1]:
                continue
            
            # Push the new state to the stack: next index, updated combination, updated sum
            stack.append((i + 1, c_omb + [pieces[i]], current_sum + pieces[i]))

    return result

pieces = [10, 1, 2, 7, 6, 1, 5]
target = 8
combination = count_combinations(pieces, target)
print(len(combination))
print(combination)
