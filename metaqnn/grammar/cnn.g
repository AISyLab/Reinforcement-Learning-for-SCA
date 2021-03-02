parser Cnn:
    ignore:           '\\s+'
    token NUM:        '[0-9]+'
    token CONV:       'C'
    token POOL:       'P'
    token SPLIT:      'S'
    token INPUT:      'I'
    token FC:         'FC'
    token DROP:       'D'
    token GLOBALAVE:  'GAP'
    token FLATTEN:    'FLAT'
    token NIN:        'NIN'
    token BATCHNORM:  'BN'
    token SOFTMAX:    'SM'

    rule input: INPUT   {{ result = ['input'] }}
                numlist {{ return result + numlist}}

    rule layers: conv   {{return conv}}
                | nin   {{return nin}}
                | gap   {{return gap}}
                | flat  {{return flat}}
                | bn    {{return bn}}
                | pool  {{return pool}}
                | split {{return split}}
                | fc    {{return fc}}
                | drop  {{return drop}}
                | softmax {{return softmax}}


    rule conv: CONV     {{ result = ['conv'] }}
               numlist  {{ return result + numlist}}

    rule nin: NIN       {{ result = ['nin'] }}
              numlist   {{ return result + numlist}}

    rule gap: GLOBALAVE {{ result = ['gap'] }}
              numlist   {{ return result + numlist }}

    rule flat: FLATTEN  {{ result = ['flat'] }}
               numlist  {{ return result + numlist }}

    rule bn: BATCHNORM  {{ return ['bn'] }}

    rule pool: POOL     {{ result = ['pool'] }}
               numlist  {{ return result + numlist}}

    rule fc: FC         {{ result = ['fc'] }}
             numlist    {{ return result + numlist}}

    rule drop: DROP        {{ result = ['dropout']}}
               numlist  {{ return result + numlist}}

    rule softmax: SOFTMAX {{ result = ['softmax'] }}
                  numlist {{ return result + numlist }}

    rule split: SPLIT "\\{"   {{ result = ['split']}}
                net        {{ result.append(net) }}
                ("," net   {{ result.append(net) }} )*
                "\\}"         {{ return result }}

    rule numlist: "\\("     {{ result = [] }}
                  NUM       {{ result.append(int(NUM)) }}
                  ( "," NUM {{ result.append(int(NUM)) }})*
                  "\\)"     {{ return result }}

    rule net: "\\["         {{ result = [] }}
              input         {{ result.append(input)  }}
              ("," layers   {{ result.append(layers) }} )*
              "\\]"         {{ return result }}
