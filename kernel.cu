#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <string.h>
#include "string"
#include <cmath>

#include <chrono>
#include <iostream> 
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#define LIN 501
#define MAX_LIN_DIC 200;
#define COL_NUM_DATA 14
#define COL_CAT_DATA 11

typedef struct {
    int id;
    char d[256];
} dado;

dado numericos[LIN][COL_NUM_DATA];
dado categoricos[LIN][COL_CAT_DATA];

typedef struct {
    int id;
    char d[256];
} mapa;

mapa dicDados[200][COL_CAT_DATA];


using namespace std;

vector<string> nomesArquivos = { "cdtup.csv", "berco.csv", "portoatracacao.csv", "mes.csv", "tipooperacao.csv",
        "tiponavegacaoatracacao.csv", "terminal.csv", "origem.csv", "destino.csv", "naturezacarga.csv", "sentido.csv" };
map<int, string> idxColuna;

fstream arquivoPrincipal;

int NUM_LINHAS_LIDAS = 0;

bool fimDoArq = true;

void criarMapComNomeDaColunaAndPosicao() {
    vector<int> numeroDaColuna = { 0, 1, 2, 4, 5, 6, 7, 16, 17, 19, 22 };
    //788883, BRCDO, 101, Cabedelo, 2016, ago, Marinha, "Apoio Portuário", "Cais Público", 0, 0, 0, 20, 0, 20, 20, 17880751, BRIQI, BRCDO, 2710, "Granel Líquido e Gasoso", 0, 2000, Desembarcados, "", 0
    int numColum = 0;
    for (int i = 0; i < nomesArquivos.size(); ++i) {

        idxColuna.insert(pair<int, string>(numeroDaColuna[numColum], nomesArquivos[i]));
        numColum++;
    }

}


int geraMatriz() {

    int colunaNumerica = 0;
    int colunaCategorica = 0;

    int numLinhas;
    std::string valor;
    for (numLinhas = 0; numLinhas < LIN; numLinhas++) {
        for (int numColum = 0; numColum < (COL_CAT_DATA + COL_NUM_DATA); numColum++) {

            if (!getline(arquivoPrincipal, valor, ',')) {
                fimDoArq = false;
                return numLinhas;
            };

            if (idxColuna.find(numColum) == idxColuna.end()) {
                memset(numericos[numLinhas][colunaNumerica].d, 0, sizeof(numericos[numLinhas][colunaNumerica].d));
                strncpy(numericos[numLinhas][colunaNumerica].d, valor.c_str(), valor.size());

                colunaNumerica++;

            }
            else
            {
                memset(categoricos[numLinhas][colunaCategorica].d, 0, sizeof(categoricos[numLinhas][colunaCategorica].d));
                strncpy(categoricos[numLinhas][colunaCategorica].d, valor.c_str(), valor.size());



                colunaCategorica++;
            }
            valor.clear();
        }
        colunaNumerica = 0;
        colunaCategorica = 0;
    }
    return numLinhas;
}

void exportaMatriz(int maxlinhas);

void limpaArquivo() {
    for (int i = 0; i < nomesArquivos.size(); ++i) {
        fstream arq;
        arq.open(nomesArquivos[i], fstream::out);
        // arq.clear();
        arq.close();
    }
};
   

void linhaInicial() {
    string linha;
    fstream arquivo;
    arquivo.open("saida.csv", fstream::app);
    
    getline(arquivoPrincipal, linha);

    arquivo << linha << endl;

    getline(arquivoPrincipal, linha, ',');

    arquivo << linha << ',';

    arquivo.close();

}
/*
__device__  bool cudastrcmp(char s1[256], char s2[256]) {
    int posChar = 0;
    while (s1[posChar] == s2[posChar]) {
        if (s1[posChar] == '\0' && s2[posChar] == '\0') {
            return true;
        }
        posChar++;
    }
    return false;
}

__global__ 
void criacaoDeDicionario(dado *mainDados, mapa* dicDados, int lin, int col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (col * 200)) {
     
        // busca no dicionario
        for (int varreDicio = 0; varreDicio < (col * 200); varreDicio++) {
            
            bool trocaNoDicionario = cudastrcmp(dicDados[varreDicio].d, mainDados[i].d) && dicDados[varreDicio].numDaMenorLinha > i || dicDados[varreDicio].d[0] == '\0';

            if(trocaNoDicionario) {
                // escrita no dicionario
                int posChar = 0;
                for (posChar; mainDados[i].d[posChar] != '\0'; posChar++) {
                    dicDados[varreDicio].d[posChar] = mainDados[i].d[posChar];
                }
                dicDados[varreDicio].d[posChar] = '\0';
                dicDados[varreDicio].numDaMenorLinha = i;
                break;
            }
        }
    }
}
*/
__global__
void insercaoDeDados(dado* mainDados, mapa* dicDados, int lin, int col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < (lin*col)) {
        mainDados[index].id = 2;
    }

}



int main()
{
    auto start = std::chrono::steady_clock::now();

    criarMapComNomeDaColunaAndPosicao();

    arquivoPrincipal.open("dataset_00_1000_sem_virg.csv", fstream::in);

    if (arquivoPrincipal.is_open() == false) {
        perror("Erro ao abrir o arquivo");
        return 1;
    }

    linhaInicial();
   
    dado * d_categoricos, * d_newCategoricos;
    mapa * d_dicDados;

    /*
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < COL_CAT_DATA; j++) {
            strcpy(dicDados[i + j].d, "\0");
            dicDados[i + j].id = -1;
        }
    }
    */
    int QNTD_LINHAS_LIDAS;
    while(fimDoArq) {
        
        QNTD_LINHAS_LIDAS = geraMatriz();
        cudaMalloc((void**)&d_categoricos, sizeof(dado) * LIN * COL_CAT_DATA);
        
        cudaMalloc((void**)&d_dicDados, sizeof(mapa) * COL_CAT_DATA * 200);
        
        cudaMemcpy(d_categoricos, categoricos, sizeof(dado) * LIN * COL_CAT_DATA, cudaMemcpyHostToDevice);
        
        cudaMemcpy(d_dicDados, dicDados, sizeof(mapa) * COL_CAT_DATA * 200, cudaMemcpyHostToDevice);
      
        insercaoDeDados << <COL_CAT_DATA, LIN >> > (d_categoricos, d_dicDados, QNTD_LINHAS_LIDAS, COL_CAT_DATA);

        cudaMemcpy(categoricos, d_categoricos, sizeof(mapa) * COL_CAT_DATA * 200, cudaMemcpyDeviceToHost);

        printf("%s %d\n", categoricos[0][0].d, categoricos[0][0].id);
        
        exportaMatriz(QNTD_LINHAS_LIDAS);
        cudaFree(d_categoricos);
        cudaFree(d_dicDados);
       // free(categoricos);
    }

    arquivoPrincipal.close();

    auto end = chrono::steady_clock::now();
    std::cout << "Tempo       : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;    // Ending of parallel region 
}




//TROCAR DADOS NUMÉRICOS PARA CONVERSÃO DE INTEIRO PARA STRING
void exportaMatriz(int maxLinhas) {
    int colunaNumerica = 0;
    int colunaCategorica = 0;
    fstream saida;
    saida.open("saida.csv", fstream::app);

    for (int i = 0; i < maxLinhas; i++) {
        for (int j = 0; j < (COL_NUM_DATA + COL_CAT_DATA); j++) {
            if (idxColuna.find(j) != idxColuna.end()) {
                saida << categoricos[i][colunaCategorica].d << ',';
                // printf("%s\n", categoricos[i][colunaCategorica].d);
                colunaCategorica++;
            }
            else
            {
                saida << numericos[i][colunaNumerica].d << ',';
                if(i == 0) printf("%d %s\n", j, numericos[i][colunaNumerica].d);
                colunaNumerica++;

            }
        }
        colunaNumerica = 0;
        colunaCategorica = 0;
    }
    /*
    if (fimDoArq) {
        saida << "0\n";
    }
    */

    saida.close();
}
