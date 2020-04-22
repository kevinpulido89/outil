def describe(element):
    print()
    t = type(element)
    
    try:
        return print('Type:',t,'|| Size:', element.shape)
    except:
        return print('Type:',t,'|| Size:', len(element))
    finally:
        print()

def concatenar_csv(path):
    from os import listdir
    import pandas

    files = [file for file in listdir(path) if not file.startswith('.')] # Ignore hidden files

    full_file = pandas.DataFrame()

    for file in files:
        df = pandas.read_csv(path+"/"+file)
        full_file = pandas.concat([full_file,df])

    return full_file
