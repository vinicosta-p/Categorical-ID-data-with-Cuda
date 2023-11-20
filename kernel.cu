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

#define LIN 1002
#define MAX_LIN_DIC 200;
#define COL_NUM_DATA 15
#define COL_CAT_DATA 11

typedef struct {
    char d[256];
} dado;

dado numericos[LIN][COL_NUM_DATA];
dado categoricos[LIN][COL_CAT_DATA];

typedef struct {
    int id;
    int numDaMenorLinha;
    char d[256];
} mapa;


using namespace std;

vector<string> nomesArquivos = { "cdtup.csv", "berco.csv", "portoatracacao.csv", "mes.csv", "tipooperacao.csv",
        "tiponavegacaoatracacao.csv", "terminal.csv", "origem.csv", "destino.csv", "naturezacarga.csv", "sentido.csv" };
map<int, string> idxColuna;

fstream arquivoPrincipal;

int NUM_LINHAS_LIDAS = 0;

bool fimDoArq = true;

void criarMapComNomeDaColunaAndPosicao();

int geraMatriz();

void exportaMatriz();

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

    getline(arquivoPrincipal, linha);

    arquivo.open("saida.csv", fstream::app);

    arquivo << linha << endl;

    arquivo.close();

}

__global__ 
void criacaoDeDicionario(dado *mainDados, mapa* dicDados, int lin, int col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (col * 200)) {
        int posChar = 0;
        for (posChar; mainDados[i].d[posChar] != '\0'; posChar++) {
            dicDados[i].d[posChar] = mainDados[i].d[posChar];
        }
        dicDados[i].d[posChar] = '\0';
        dicDados[i].numDaMenorLinha = i;
        
    }
}
/*
__global__
void insercaoDeDados(dado* mainDados, mapa* copyDados, int lin, int col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < COL_CAT_DATA) {
        copyDados[i] = mainDados[i];
    }
    mainDados[i].d;

}
*/


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
   
    dado * d_categoricos;
    mapa * dicDados, * d_dicDados;

    dicDados = (mapa*)malloc(sizeof(mapa) * COL_CAT_DATA * 200);

    while(fimDoArq) {

        geraMatriz();

        cudaMalloc((void**)&d_categoricos, sizeof(dado) * LIN * COL_CAT_DATA);
        cudaMalloc((void**)&d_dicDados, sizeof(mapa) * COL_CAT_DATA * 200);

        cudaMemcpy(d_categoricos, categoricos, sizeof(dado) * LIN * COL_CAT_DATA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dicDados, dicDados, sizeof(mapa) * COL_CAT_DATA * 200, cudaMemcpyHostToDevice);

        criacaoDeDicionario << <COL_CAT_DATA, LIN >> > (d_categoricos, d_dicDados, LIN, COL_CAT_DATA);

        cudaMemcpy(dicDados, d_dicDados, sizeof(mapa) * COL_CAT_DATA * 200, cudaMemcpyDeviceToHost);

        printf("%d\n", dicDados[0].numDaMenorLinha);
       
        cudaFree(d_categoricos);
        cudaFree(d_dicDados);
    }
    
    exportaMatriz();

    arquivoPrincipal.close();

    auto end = chrono::steady_clock::now();
    std::cout << "Tempo       : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;    // Ending of parallel region 
}


void criarMapComNomeDaColunaAndPosicao() {
    vector<int> numeroDaColuna = { 1, 2, 3, 5, 6, 7, 8, 17, 18, 20, 23 };
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
            
            valor.append(",");

            if(idxColuna.find(numColum) == idxColuna.end()){
                strncpy(numericos[numLinhas][colunaNumerica].d, valor.c_str(), valor.size());
                
                colunaNumerica++;
                
            }
            else
            {
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

void exportaMatriz() {
    int colunaNumerica = 0;
    int colunaCategorica = 0;
    fstream saida;
    saida.open("saida.csv", fstream::app);

    for (int i = 0; i < LIN; i++) {
        for (int j = 0; j < (COL_NUM_DATA + COL_CAT_DATA); j++) {
            if (idxColuna.find(j) == idxColuna.end()) {
                saida << numericos[i][colunaNumerica].d;
                //if (i == 0) printf("%s\n", numericos[i][colunaNumerica].d);
                colunaNumerica++;
            }
            else
            {
                saida << categoricos[i][colunaCategorica].d;
                //if (i == 0) printf("%s\n", categoricos[i][colunaCategorica].d);
                colunaCategorica++;

            }
        }
        colunaNumerica = 0;
        colunaCategorica = 0;
    }
    if (fimDoArq) {
        saida << "0\n";
    }

    saida.close();
}
