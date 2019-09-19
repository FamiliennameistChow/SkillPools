list_a = [4, 2, 7, 6]
#
# sum = 0
# for i in range(len(list_a) - 1):
#     sum = abs(list_a[i] - list_a[i+1])
#
# list_a.sort(reverse=True)
# print(list_a)
count = 0
while(len(list_a) <= 2):
    for i in range(1, len(list_a) - 1):
        if (list_a[i] < list_a[i+1]) and (list_a[i] > list_a[i-1]):
            count += 1
            list_a.remove(list_a[i])



import sys

result = []
for line in sys.stdin:
    data = list(map(int, line.strip().split(",")))
    length = len(data)
    number = 0
    for i in range(length - 1):
        if data[i + 1] >= data[i]:
            continue
        else:
            data[i], data[i + 1] = data[i + 1], data[i]
            number += 1
            for j in range(i, 0, -1):
                if data[j] < data[j - 1]:
                    data[j], data[j - 1] = data[j - 1], data[j]
                    number += 1
                else:
                    break
    ##    print(data)
    result.append(number)

print("\n".join(map(str, result)))
