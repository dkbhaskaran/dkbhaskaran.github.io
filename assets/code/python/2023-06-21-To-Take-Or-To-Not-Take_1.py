def combinationSumI(source: list, targetSum: int,
                    combination: list, index: int) -> None:
    # Calculate the current sum of the combination
    currSum = sum(combination)

    # Base case: if index is out of range or current sum
    # exceeds target, return
    if index >= len(source) or currSum > targetSum:
        return

    # If we reach the target sum, print the combination
    if currSum == targetSum:
        print(combination)
        return

    # Include the current element and recurse
    combination.append(source[index])
    combinationSumI(source, targetSum, combination, index)

    # Backtrack: remove the last element
    combination.pop()

    # Don't take the element at the current index and move
    # to the next index
    combinationSumI(source, targetSum, combination,
                    index + 1)

# Example call
combinationSumI([2, 3, 5], 8, [], 0)
