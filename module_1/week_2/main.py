# Part I - Ex1
def max_kernel(num_list, k):
    if num_list is None:
        return None
    if len(num_list) < k:
        return [max(num_list)]
    result = []
    p1 = 0
    p2 = k - 1
    while p2 < len(num_list):
        current_max = max(num_list[p1:p2+1])
        p1 += 1
        p2 += 1
        result.append(current_max)
    return result


# Part I - Ex2
def count_chars(s):
    dictionary = {}
    for c in s:
        if c in dictionary:
            dictionary[c] += 1
        else:
            dictionary[c] = 1
    return dict(sorted(dictionary.items()))


# Part I - Ex3
def count_word(file_path):
    dictionary = {}
    with open(file_path, 'r') as f:
        content = f.read()
        words = content.lower().split()
        for word in words:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1
    return dict(sorted(dictionary.items()))


# Part I - Ex4
def calculate_levenshtein_distance(s1, s2):
    # Step 1: Initialize matrix D
    m = len(s1) + 1
    n = len(s2) + 1
    D = [[0 for _ in range(n)] for _ in range(m)]
    # Step 2: Initialize first row and first column
    D[0] = [i for i in range(n)]
    for i in range(m):
        D[i][0] = i
    # Step 3: Fill the whole matrix
    for i in range(1, m):
        for j in range(1, n):
            if s1[i-1] == s2[j-1]:
                D[i][j] = D[i-1][j-1]
            else:
                # Insert, Delete, Substitute accordingly
                D[i][j] = min(D[i-1][j] + 1,
                              D[i][j-1] + 1,
                              D[i-1][j-1] + 1)
    # Step 4: Return the last element which is the distance
    return D[-1][-1]


# Part II - Ex5
def check_the_number(N):
    list_of_numbers = []
    result = ''
    for i in range(1, 5):
        list_of_numbers.append(i)
    if N in list_of_numbers:
        result = 'True'
    if N not in list_of_numbers:
        result = 'False'
    return result


# Part II - Ex6
def my_function_6(data, max_n, min_n):
    result = []
    for i in data:
        if i < min_n:
            result.append(min_n)
        elif i > max_n:
            result.append(max_n)
        else:
            result.append(i)
    return result


# Part II - Ex7
def my_function_7(x, y):
    x.extend(y)
    return x


# Part II - Ex8
def my_function_8(n_list):
    return min(n_list)


# Part II - Ex9
def my_function_9(n_list):
    return max(n_list)


# Part II - Ex10
def my_function_10(intergers, number=1):
    return any([True if i == number else False for i in intergers])


# Part II - Ex11
def my_function_11(list_nums=[0, 1, 2]):
    var = 0
    for i in list_nums:
        var += i
    return var / len(list_nums)


# Part II - Ex12
def my_function_12(data):
    var = []
    for i in data:
        if i % 3 == 0:
            var.append(i)
    return var


# Part II - Ex13
def my_function_13(y):
    var = 1
    while y > 1:
        var *= y
        y -= 1
    return var


# Part II - Ex14
def my_function_14(x):
    return x[::-1]


# Part II - Ex15
def function_helper_15(x):
    return 'T' if x > 0 else 'N'


def my_function_15(data):
    res = [function_helper_15(x) for x in data]
    return res


# Part II - Ex16
def function_helper_16(x, data):
    for i in data:
        if x == i:
            return 0
    return 1


def my_function_16(data):
    res = []
    for i in data:
        if function_helper_16(i, res):
            res.append(i)
    return res


if __name__ == '__main__':
    print('Part I - Ex1')
    n_l = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
    print(max_kernel(n_l, 3))
    print('-------------------')

    print('Part I - Ex2')
    s1 = 'Happiness'
    print(count_chars(s1))
    s2 = 'smiles'
    print(count_chars(s2))
    print('-------------------')

    print('Part I - Ex3')
    print(count_word('./P1_data.txt'))
    print('-------------------')

    print('Part I - Ex4')
    s1 = 'yu'
    s2 = 'you'
    print(f'Levenshtein distance between {s1} and {
          s2} is: {calculate_levenshtein_distance(s1, s2)}')
    print('-------------------')

    print('Part II - Ex1')
    assert max_kernel([3, 4, 5, 1, -44], 3) == [5, 5, 5]
    num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
    k = 3
    print(max_kernel(num_list, k))
    print('-------------------')

    print('Part II - Ex2')
    assert count_chars('Baby') == {'B': 1, 'a': 1, 'b': 1, 'y': 1}
    print(count_chars('smiles'))
    print('-------------------')

    print('Part II - Ex3')
    file_path = './P1_data.txt'
    result = count_word(file_path)
    assert result['who'] == 3
    print(result['man'])
    print('-------------------')

    print('Part II - Ex4')
    assert calculate_levenshtein_distance('hi', 'hello') == 4.0
    print(calculate_levenshtein_distance('hola', 'hello'))
    print('-------------------')

    print('Part II - Ex5')
    N = 7
    assert check_the_number(N) == 'False'
    N = 2
    result = check_the_number(N)
    print(result)
    print('-------------------')

    print('Part II - Ex6')
    my_list = [5, 2, 5, 0, 1]
    max_n = 1
    min_n = 0
    assert my_function_6(max_n=max_n, min_n=min_n,
                         data=my_list) == [1, 1, 1, 0, 1]
    my_list = [10, 2, 5, 0, 1]
    max_n = 2
    min_n = 1
    print(my_function_6(max_n=max_n, min_n=min_n, data=my_list))
    print('-------------------')

    print('Part II - Ex7')
    list_num1 = ['a', 2, 5]
    list_num2 = [1, 1]
    list_num3 = [0, 0]
    assert my_function_7(list_num1, my_function_7(
        list_num2, list_num3)) == ['a', 2, 5, 1, 1, 0, 0]
    list_num1 = [1, 2]
    list_num2 = [3, 4]
    list_num3 = [0, 0]
    print(my_function_7(list_num1, my_function_7(list_num2, list_num3)))
    print('-------------------')

    print('Part II - Ex8')
    my_list = [1, 22, 93, -100]
    assert my_function_8(my_list) == -100
    my_list = [1, 2, 3, -1]
    print(my_function_8(my_list))
    print('-------------------')

    print('Part II - Ex9')
    my_list = [1001, 9, 100, 0]
    assert my_function_9(my_list) == 1001
    my_list = [1, 9, 9, 0]
    print(my_function_9(my_list))
    print('-------------------')

    print('Part II - Ex10')
    my_list = [1, 3, 9, 4]
    assert my_function_10(my_list, -1) == False
    my_list = [1, 2, 3, 4]
    print(my_function_10(my_list, 2))
    print('-------------------')

    print('Part II - Ex11')
    assert my_function_11([4, 6, 8]) == 6
    print(my_function_11())
    print('-------------------')

    print('Part II - Ex12')
    assert my_function_12([3, 9, 4, 5]) == [3, 9]
    print(my_function_12([1, 2, 3, 5, 6]))
    print('-------------------')

    print('Part II - Ex13')
    assert my_function_13(8) == 40320
    print(my_function_13(4))
    print('-------------------')

    print('Part II - Ex14')
    x = 'I can do it'
    assert my_function_14(x) == 'ti od nac I'
    x = 'apricot'
    print(my_function_14(x))
    print('-------------------')

    print('Part II - Ex15')
    data = [10, 0, -10, -1]
    assert my_function_15(data) == ['T', 'N', 'N', 'N']
    data = [2, 3, 5, -1]
    print(my_function_15(data))
    print('-------------------')

    print('Part II - Ex16')
    lst = [10, 10, 9, 7, 7]
    assert my_function_16(lst) == [10, 9, 7]
    lst = [9, 9, 8, 1, 1]
    print(my_function_16(lst))
    print('-------------------')
