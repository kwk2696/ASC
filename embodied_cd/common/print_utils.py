
def print_error(*message): # red
    print('\033[91m', *message, '\033[0m')

def print_warn(*message): # yellow
    print('\033[93m', *message, '\033[0m')

def print_check(*message): # blue
    print('\033[96m', *message, '\033[0m')

def print_pass(*message): # green
    print('\033[92m', *message, '\033[0m')
