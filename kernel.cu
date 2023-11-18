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

#define LIN 1001
#define COL 26

typedef struct {
    char d[50];
} dado;

dado matrizDeDados[LIN][COL];

using namespace std;

vector<string> nomesArquivos = { "cdtup.csv", "berco.csv", "portoatracacao.csv", "mes.csv", "tipooperacao.csv",
        "tiponavegacaoatracacao.csv", "terminal.csv", "origem.csv", "destino.csv", "naturezacarga.csv", "sentido.csv" };
map<string, int> idxColuna;

int NUM_LINHAS_LIDAS = 0;

bool fimDoArq = false;

char* ReadFile(const char* filename)
{
    char* buffer = NULL;
    int string_size, read_size;
    FILE* handler = fopen(filename, "r");

    if (handler)
    {
        fseek(handler, 0, SEEK_END);
        string_size = ftell(handler);
        rewind(handler);
        buffer = (char*)malloc(sizeof(char) * (string_size + 1));
        read_size = fread(buffer, sizeof(char), string_size, handler);
        buffer[string_size] = '\0';

        if (string_size != read_size)
        {
            free(buffer);
            buffer = NULL;
        }

        fclose(handler);
    }

    return buffer;
}

void criarMapComNomeDaColunaAndPosicao();

void geraMatriz(FILE*, dado[][COL]);

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
    if (i < LIN + COL) copyDados[i] = mainDados[i];
}

int main()
{
    auto start = std::chrono::steady_clock::now();

    criarMapComNomeDaColunaAndPosicao();

    linhaInicial();

    FILE* arquivo;

    arquivo = fopen("dataset_00_1000_sem_virg.csv", "r");

    if (arquivo == NULL) {
        perror("Erro ao abrir o arquivo");
        return 1;
    }
    geraMatriz(arquivo, matrizDeDados);

    dado * d_matrizDeDados, *d_copyDados;
    dado * copyDados;

    copyDados = (dado*)malloc(sizeof(dado) * LIN * COL);

    cudaMalloc((void**)&d_matrizDeDados, sizeof(dado) * LIN * COL);
    cudaMalloc((void**)&d_copyDados, sizeof(dado) * LIN * COL);
    cudaMemcpy(d_matrizDeDados, matrizDeDados, sizeof(dado) * LIN * COL, cudaMemcpyHostToDevice);
    parallelCodeAndKey << <COL, LIN>> > (d_matrizDeDados, d_copyDados, LIN, COL);
    cudaMemcpy(copyDados, d_copyDados, sizeof(dado) * LIN * COL, cudaMemcpyDeviceToHost);
    printf("%s\n", copyDados[26]);
    exportaMatriz(matrizDeDados);

    fclose(arquivo);

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

void geraMatriz(FILE* arquivo, dado matriz[][COL]) {
    char string[320];
    int i = 0;
    while (fgets(string, sizeof(string), arquivo) != NULL) {
        char* virgula = 0;
        for (int j = 0; j < COL; j++) {
            if ((virgula = strchr(string, ',')) == NULL) {
                if ((virgula = strchr(string, '\n')) == NULL) {
                    virgula = strchr(string, '\0');
                }
            }
            size_t tamanho = virgula - string;
            strncpy(matriz[i][j].d, string, tamanho);
            matriz[i][j].d[tamanho] = '\0';
            strcpy(string, virgula + 1);
        }
        i++;
    }
}

void exportaMatriz(dado matriz[][COL]) {
    FILE* saida = fopen("saida.csv", "w");
    for (int i = 0; i < LIN; i++) {
        for (int j = 0; j < COL; j++) {
            fprintf(saida, "%s", matriz[i][j].d);
            if (j + 1 != COL) {
                fprintf(saida, ",");
            }

        }
        fprintf(saida, "\n");
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