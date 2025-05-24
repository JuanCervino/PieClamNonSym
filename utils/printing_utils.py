from inspect import currentframe, getframeinfo

def filename_n_line():
    cf = currentframe()
    #DAN: f_back is the function that called this
    filename = getframeinfo(cf.f_back).filename
    linum = cf.f_back.f_lineno
    print(filename, linum)


def filename_n_line_str():
    cf = currentframe()
    #DAN: f_back is the function that called this
    filename = getframeinfo(cf.f_back).filename
    linum = cf.f_back.f_lineno
    return filename + ':' + str(linum) + '    '


def printd(msg):
    cf = currentframe()
    #DAN: f_back is the function that called this
    filename = getframeinfo(cf.f_back).filename
    linum = cf.f_back.f_lineno
    print('\n')
    print(filename + ':' + str(linum) + f':::  \n{msg}')
    


def bp():
    cf = currentframe()
    #DAN: f_back is the function that called this
    filename = getframeinfo(cf.f_back).filename
    linum = cf.f_back.f_lineno
    print(filename, linum)
    while True:
        inpt = input(f'debug stuff \nVVVVVVVVV \n')
        if inpt == 'fin':
            break
        try:
            eval('print(' + inpt + ')')
        except Exception:
            print('Invalid Input: ' + inpt)
            continue


def print_dolphin():
    print('''
             ,-._
           _.-'  '--.
         .'      _  -`\_
        / .----.`_.'----'
        ;/     `
 jgs   /_;
    ''')


def print_escher():
    print('''
              .
           .-` :-
        .-` .-` :
     .-` .-` .: :
  .-` .-` .-: : :
-:  -: .-`  : : :
: `-. `-.   : : :
'-.  `-. '-.: : :
   `-.  `-. : : :
      `-.  `- : :
         `-.  : :
            `-:-`  mic
    ''')


def print_end_fit():
    print('''
 888888 88b 88 8888b.      888888 88 888888     88""Yb  dP"Yb  88   88 88b 88 8888b.  
 88__   88Yb88  8I  Yb     88__   88   88       88__dP dP   Yb 88   88 88Yb88  8I  Yb 
 88""   88 Y88  8I  dY     88""   88   88       88"Yb  Yb   dP Y8   8P 88 Y88  8I  dY 
 888888 88  Y8 8888Y"      88     88   88       88  Yb  YbodP  `YbodP' 88  Y8 8888Y"    
        ''')