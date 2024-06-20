import torch
from torch import nn
from abc import ABC, abstractmethod


# Part I - Ex1
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        return torch.exp(data) / torch.sum(torch.exp(data))


class SoftmaxStable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        x_max = torch.max(data, dim=0, keepdim=True)
        x_exp = torch.exp(data - x_max.values)
        partition = x_exp.sum(0, keepdim=True)
        return x_exp / partition

# Part I - Ex2


class Ward():
    def __init__(self, name):
        self.__name = name
        self.__people = []

    def add_person(self, person):
        self.__people.append(person)

    def describe(self):
        print(f'Ward Name: {self.__name}')
        for person in self.__people:
            person.describe()

    def count_doctor(self):
        n_doctors = 0
        for person in self.__people:
            if isinstance(person, Doctor):
                n_doctors += 1
        return n_doctors

    def sort_age(self):
        self.__people.sort(key=lambda x: x.get_yob(), reverse=True)

    def compute_average(self):
        sum_yob = 0
        n_teachers = 0
        for person in self.__people:
            if isinstance(person, Teacher):
                sum_yob += person.get_yob()
                n_teachers += 1
        if n_teachers == 0:
            print('No teacher in the ward')
        else:
            return sum_yob / n_teachers


class Person(ABC):
    def __init__(self, name, yob):
        self._name = name
        self._yob = yob

    def get_yob(self):
        return self._yob

    @abstractmethod
    def describe(self):
        pass


class Student(Person):
    def __init__(self, name, yob, grade):
        super().__init__(name, yob)
        self._grade = grade

    def describe(self):
        print(
            f'Student - Name: {self._name} - YoB: {self._yob} - Grade: {self._grade}')


class Teacher(Person):
    def __init__(self, name, yob, subject):
        super().__init__(name, yob)
        self._subject = subject

    def describe(self):
        print(
            f'Teacher - Name: {self._name} - YoB: {self._yob} - Subject: {self._subject}')


class Doctor(Person):
    def __init__(self, name, yob, specialist):
        super().__init__(name, yob)
        self._specialist = specialist

    def describe(self):
        print(
            f'Doctor - Name: {self._name} - YoB: {self._yob} - Specialist: {self._specialist}')


# Part I - Ex3
class MyStack():
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__stack = []

    def is_empty(self):
        return len(self.__stack) == 0

    def is_full(self):
        return len(self.__stack) == self.__capacity

    def pop(self):
        if self.is_empty():
            return None
        return self.__stack.pop()

    def push(self, value):
        if self.is_full():
            return None
        self.__stack.append(value)

    def top(self):
        if self.is_empty():
            return None
        return self.__stack[-1]


class MyQueue():
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__queue = []

    def is_empty(self):
        return len(self.__queue) == 0

    def is_full(self):
        return len(self.__queue) == self.__capacity

    def dequeue(self):
        if self.is_empty():
            return None
        return self.__queue.pop(0)

    def enqueue(self, value):
        if self.is_full():
            return None
        self.__queue.append(value)

    def front(self):
        if self.is_empty():
            return None
        return self.__queue[0]


if __name__ == '__main__':
    print('Part I - Ex1')
    data = torch.Tensor([1, 2, 3])
    softmax = Softmax()
    output = softmax(data)
    print(output)
    softmax_stable = SoftmaxStable()
    output = softmax_stable(data)
    print(output)
    print('----------------------')
    print('Part I - Ex2')
    # 2a
    student1 = Student(name='studentA', yob=2010, grade='7')
    student1.describe()
    teacher1 = Teacher(name='teacherA', yob=1969, subject='Math')
    teacher1.describe()
    doctor1 = Doctor(name='doctorA', yob=1945, specialist='Endocrinologists')
    doctor1.describe()
    # 2b
    print()
    teacher2 = Teacher(name='teacherB', yob=1995, subject='History')
    doctor2 = Doctor(name='doctorB', yob=1975, specialist='Cardiologists')
    ward1 = Ward(name='Ward1')
    ward1.add_person(student1)
    ward1.add_person(teacher1)
    ward1.add_person(teacher2)
    ward1.add_person(doctor1)
    ward1.add_person(doctor2)
    ward1.describe()
    # 2c
    print(f'\nNumber of doctors: {ward1.count_doctor()}')
    # 2d
    print('\nAfter softing Age of Ward1 people')
    ward1.sort_age()
    ward1.describe()
    # 2e
    print(f'\nAverage year of birth (teachers): {ward1.compute_average()}')
    print('----------------------')
    print('Part I - Ex3')
    stack1 = MyStack(capacity=5)
    stack1.push(1)
    stack1.push(2)
    print(stack1.is_full())
    print(stack1.top())
    print(stack1.pop())
    print(stack1.top())
    print(stack1.pop())
    print(stack1.is_empty())
    queue1 = MyQueue(capacity=5)
    queue1.enqueue(1)
    queue1.enqueue(2)
    print(queue1.is_full())
    print(queue1.front())
    print(queue1.dequeue())
    print(queue1.front())
    print(queue1.dequeue())
    print(queue1.is_empty())
    print('----------------------')
    print('Part II - Ex1')
    data = torch.Tensor([1, 2, 3])
    softmax_function = nn.Softmax(dim=0)
    output = softmax_function(data)
    assert round(output[0].item(), 2) - 0.09 < 0.01
    print(output)
    print('----------------------')
    print('Part II - Ex2')
    data = torch.Tensor([5, 2, 4])
    my_softmax = Softmax()
    output = my_softmax(data)
    assert round(output[-1].item(), 2) - 0.26 < 0.01
    print(output)
    print('----------------------')
    print('Part II - Ex3')
    data = torch.Tensor([1, 2, 300000000])
    my_softmax = Softmax()
    output = my_softmax(data)
    assert round(output[0].item(), 2) < 0.01
    print(output)
    print('----------------------')
    print('Part II - Ex4')
    data = torch.Tensor([1, 2, 3])
    softmax_stable = SoftmaxStable()
    output = softmax_stable(data)
    assert round(output[-1].item(), 2) - 0.67 < 0.01
    print(output)
    print('----------------------')
    print('Part II - Ex5')
    student1 = Student(name='studentZ2023', yob=2011, grade='6')
    assert student1._yob == 2011
    student1.describe()
    print('----------------------')
    print('Part II - Ex6')
    teacher1 = Teacher(name='teacherZ2023', yob=1991, subject='History')
    assert teacher1._yob == 1991
    teacher1.describe()
    print('----------------------')
    print('Part II - Ex7')
    doctor1 = Doctor(name='doctorZ2023', yob=1981,
                     specialist='Endocrinologists')
    assert doctor1._yob == 1981
    doctor1.describe()
    print('----------------------')
    print('Part II - Ex8')
    student1 = Student(name='studentA', yob=2010, grade='7')
    teacher1 = Teacher(name='teacherA', yob=1969, subject='Math')
    teacher2 = Teacher(name='teacherB', yob=1995, subject='History')
    doctor1 = Doctor(name='doctorA', yob=1945, specialist='Endocrinologists')
    doctor2 = Doctor(name='doctorB', yob=1975, specialist='Cardiologists')
    ward1 = Ward(name='Ward1')
    ward1.add_person(student1)
    ward1.add_person(teacher1)
    ward1.add_person(teacher2)
    ward1.add_person(doctor1)
    ward1.add_person(doctor2)
    print(ward1.count_doctor())
    print('----------------------')
    print('Part II - Ex9')
    stack1 = MyStack(capacity=5)
    stack1.push(1)
    assert stack1.is_full() == False
    stack1.push(2)
    print(stack1.is_full())
    print('----------------------')
    print('Part II - Ex10')
    stack1 = MyStack(capacity=5)
    stack1.push(1)
    assert stack1.is_full() == False
    stack1.push(2)
    print(stack1.top())
    print('----------------------')
    print('Part II - Ex11')
    queue1 = MyQueue(capacity=5)
    queue1.enqueue(1)
    assert queue1.is_full() == False
    queue1.enqueue(2)
    print(queue1.is_full())
    print('----------------------')
    print('Part II - Ex12')
    queue1 = MyQueue(capacity=5)
    queue1.enqueue(1)
    assert queue1.is_full() == False
    queue1.enqueue(2)
    print(queue1.front())
    print('----------------------')
