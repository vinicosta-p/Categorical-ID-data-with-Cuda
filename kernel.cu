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

#define LIN 1000
#define COL 26
#define DIC 200

typedef struct {
    char d[50];
} dado;

dado matrizDeDados[LIN][COL];

typedef struct {
    int numThread;
    char d[50];
} dicionario;

dicionario dic[DIC][COL];

using namespace std;

vector<string> nomesArquivos = { "cdtup.csv", "berco.csv", "portoatracacao.csv", "mes.csv", "tipooperacao.csv",
        "tiponavegacaoatracacao.csv", "terminal.csv", "origem.csv", "destino.csv", "naturezacarga.csv", "sentido.csv" };
map<string, int> idxColuna;

fstream arquivoPrincipal;

int NUM_LINHAS_LIDAS = 0;

bool fimDoArq = false;

void criarMapComNomeDaColunaAndPosicao();

int geraMatriz(dado[][COL]);

void imprimeMatriz(dado[][COL]);

void initMatriz(dado[][COL]);

void exportaMatriz(dado[][COL]);

void limpaArquivo() {
    for (int i = 0; i < nomesArquivos.size(); ++i) {
        fstream arq;
        arq.open(nomesArquivos[i], fstream::out);
        // arq.clear();
        arq.close();
    }
};

void linhaInicial() {
    fstream arquivo;
    string linha;

    arquivo.open("dataset_00_1000_sem_virg.csv", fstream::in);

    getline(arquivo, linha);

    arquivo.close();

    arquivo.open("saida.csv", fstream::out);

    arquivo << linha;

    arquivo.close();

}

__global__ 
void parallelCodeAndKey(dado *mainDados,dado *copyDados, int lin, int col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < LIN + COL) {
        copyDados[i] = mainDados[i];
    }
    mainDados[i].d;

}



int main()
{
    auto start = std::chrono::steady_clock::now();

    criarMapComNomeDaColunaAndPosicao();

    linhaInicial();

    arquivoPrincipal.open("dataset_00_1000_sem_virg.csv", fstream::in);

    if (arquivoPrincipal.is_open() == false) {
        perror("Erro ao abrir o arquivo");
        return 1;
    }
    geraMatriz(matrizDeDados);
    /*
    dado* d_matrizDeDados, * d_copyDados;
    dado* copyDados;

    for (int i = 0; i < 1; i++) {
        
        copyDados = (dado*)malloc(sizeof(dado) * LIN * COL);
        
        cudaMalloc((void**)&d_matrizDeDados, sizeof(dado) * LIN * COL);
        cudaMalloc((void**)&d_copyDados, sizeof(dado) * LIN * COL);

        cudaMemcpy(d_matrizDeDados, matrizDeDados, sizeof(dado) * LIN * COL, cudaMemcpyHostToDevice);

        parallelCodeAndKey << <COL, LIN >> > (d_matrizDeDados, d_copyDados, LIN, COL);

        cudaMemcpy(copyDados, d_copyDados, sizeof(dado) * LIN * COL, cudaMemcpyDeviceToHost);

        //printf("%s\n", copyDados[26]);
       
        cudaFree(d_copyDados);
        cudaFree(d_matrizDeDados);
        free(copyDados);
    }
    */
    
    exportaMatriz(matrizDeDados);

    arquivoPrincipal.close();

    auto end = chrono::steady_clock::now();
    std::cout << "Tempo       : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;    // Ending of parallel region 
}


void criarMapComNomeDaColunaAndPosicao() {
    vector<int> numeroDaColuna = { 1, 2, 3, 5, 6, 7, 8, 17, 18, 20, 23 };
    int numColum = 0;
    for (int i = 0; i < nomesArquivos.size(); ++i) {
        idxColuna.insert(pair<string, int>(nomesArquivos[i], numeroDaColuna[numColum]));
        numColum++;
    }

}

int geraMatriz(dado matriz[][COL]) {

    int numLinhas;

    for (numLinhas = 0; numLinhas < LIN; numLinhas++) {
        for (int j = 0; j < COL; j++) {
            std::string valor;
            if (!getline(arquivoPrincipal, valor, ',')) {
                fimDoArq = true;
                return numLinhas;
            };
            strncpy(matriz[numLinhas][j].d, valor.c_str(), valor.size());
        }
    }
    return numLinhas;
}

void exportaMatriz(dado matriz[][COL]) {
    FILE* saida = fopen("saida.csv", "w");
    for (int i = 0; i < LIN; i++) {
        for (int j = 0; j < COL; j++) {
            fprintf(saida, "%s", matriz[i][j].d);
        }
    }
    fclose(saida);
}

void imprimeMatriz(dado matriz[][COL]) {
    for (int i = 0; i < LIN; i++) {
        printf("Linha %i: ", i + 1);
        for (int j = 0; j < COL; j++) {
            printf("%s ", matriz[i][j].d);
        }
        printf("\n\n");
    }
}

void initMatriz(dado matriz[][COL]) {
    for (int i = 0; i < LIN; i++) {
        for (int j = 0; j < COL; j++) {
            for (int k = 0; k < 50; k++) {
                matriz[i][j].d[k] = 0;
            }
        }
        printf("\n\n");
    }
}