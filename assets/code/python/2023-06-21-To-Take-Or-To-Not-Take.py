def genSubSequence(source: list, subsequence: list,
                   index: int) -> None:
    # Base case: if we've processed all elements,
    # print the current subsequence
    if index >= len(source):
        print(subsequence)
        return

    # Take the element at the current index
    subsequence.append(source[index])
    genSubSequence(source, subsequence, index + 1)

    # Backtrack by removing the last added element
    # (don't take the element)
    subsequence.pop()

    # Don't take the element at the current index
    genSubSequence(source, subsequence, index + 1)

# Example call
genSubSequence([1, 2, 3], [], 0)
