MOD_DICT = {0: -3+3j, #0000
            1: -3+1j, #0001
            2: -3-3j, #0010
            3: -3-1j, #0011
            4: -1+3j, #0100
            5: -1+1j, #0101
            6: -1-3j, #0110
            7: -1-1j, #0111
            8:  3+3j, #1000
            9:  3+1j, #1001
            10: 3-3j, #1010
            11: 3-1j, #1011
            12: 1+3j, #1100
            13: 1+1j, #1101
            14: 1-3j, #1110
            15: 1-1j} #1111

def download_file(link):
    # Nombre del archivo
    file = link.split('/')[-1]
    
    # Descarga el archivo si no se ha hecho aún
    !if ! [[ -f "$file" ]]; then wget $link; fi;
    
    # Carga los datos
    data = loadmat(file)
    
    return data

def mod_norm(const, power):
    constPow = np.mean([x**2 for x in np.abs(const)])
    scale = np.sqrt(power/constPow)
    return scale

def sync_signals(short_signal, long_signal):
    # Se concatena para asegurar de que el array recibido esté contenido dentro
    # del largo.
    longer_signal = np.concatenate((long_signal, long_signal))
    correlation = np.correlate(short_signal, longer_signal, mode = "full")
    index = np.argmax(correlation) + 1
    delay = index - len(short_signal)
    print(f"El retraso es de {delay} posiciones")

    sync_signal = np.roll(longer_signal, delay)
    sync_signal = sync_signal[:len(short_signal)]

    return short_signal, sync_signal
