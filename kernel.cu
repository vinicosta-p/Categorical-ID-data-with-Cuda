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
#include <math.h>

#include <chrono>
#include <iostream> 
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#define LIN 100000
#define COL_NUM_DATA 14
#define COL_CAT_DATA 11
#define MAX_LIN_DICIONARIO 2700

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

mapa dicDados[MAX_LIN_DICIONARIO][COL_CAT_DATA];

using namespace std;

vector<string> nomesArquivos = { "cdtup.csv", "berco.csv", "portoatracacao.csv", "mes.csv", "tipooperacao.csv","tiponavegacaoatracacao.csv", "terminal.csv", "origem.csv", "destino.csv", "naturezacarga.csv", "sentido.csv" };

map<int, string> idxColuna;

fstream arquivoPrincipal;

int NUM_LINHAS_LIDAS = 0;

bool fimDoArq = true;

vector<map<string, int>> buscaRapidaDeDado;

void pairCodigoDescricao(string nomeArquivo, int indexDaColuna, int numLinhasLidas);

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

__device__  
bool cudastrcmp(char s1[256], char s2[256]) {
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
void insercaoDeDados(dado* d_categoricos, mapa* d_dicDados, int divisionTask, int numLinhasLidas, int totalCol) {
    int TID = blockIdx.x * blockDim.x + threadIdx.x; //200

    int inicioDaLinha = TID * divisionTask * totalCol;

    if (inicioDaLinha < (numLinhasLidas*totalCol)) { // talvez seja maior ou igual

        int numLinha = inicioDaLinha;

        int contadorDeIteracoes = 0;
        
        while (numLinha < (numLinhasLidas*totalCol) && contadorDeIteracoes < divisionTask) {
            
            for (int valorDaColunaAtual = 0; valorDaColunaAtual < totalCol; valorDaColunaAtual++) {

                int indexDoDado = numLinha + valorDaColunaAtual;
                
                for (int i = 0; i < MAX_LIN_DICIONARIO; i++) {
                    
                    int posDoValorDicionario = valorDaColunaAtual + (totalCol * i);
                    
                    if (d_dicDados[posDoValorDicionario].id == 0) {
                        break;
                    }

                    if (cudastrcmp(d_dicDados[posDoValorDicionario].d, d_categoricos[indexDoDado].d)) {
                        d_categoricos[indexDoDado].id = d_dicDados[posDoValorDicionario].id;
                        break;
                    }
                }
            }
            numLinha += totalCol;
            contadorDeIteracoes++;
        }

    }

}

void inicializaMatriz_buscaRapidaDeDado() {
    for (int i = 0; i < nomesArquivos.size(); ++i) {
        map<string, int> aux; //= { nomesArquivos[i], "0" };
        buscaRapidaDeDado.push_back(aux);
    }
}

int getNumBlock(int numLinhasLidas) {
    //128 é numero de threads por bloco escolhido
    int numBlock = ceil(numLinhasLidas / 128);
    
    if (numLinhasLidas % 128 != 0) {
        numBlock++;
    }

    if (numBlock > 15) {
        numBlock = 15;
    }
    
    return numBlock;
}

int getDivisionTask(int numLinhasLidas, int cudaCore) {
    int divisionTask = 0;

    if (numLinhasLidas > cudaCore) {
        
        divisionTask = ceil(numLinhasLidas / cudaCore); // No mínimo 2
        
        if (numLinhasLidas % cudaCore != 0) {
            divisionTask++;
        }
    }
    else {
        divisionTask = 1;
    }

    return divisionTask;
}



int main()
{
    auto start = std::chrono::steady_clock::now();
     
    limpaArquivo();
    
    criarMapComNomeDaColunaAndPosicao();

    arquivoPrincipal.open("/content/drive/MyDrive/PPC2/dataset_00_sem_virg.csv", fstream::in);

    if (arquivoPrincipal.is_open() == false) {
        perror("Erro ao abrir o arquivo");
        return 1;
    }

    linhaInicial();

    inicializaMatriz_buscaRapidaDeDado();
    

   
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
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 0 representa o ID do dispositivo, que pode variar se você tiver várias GPUs.
    int cudaCore = prop.multiProcessorCount * 64;

    int QNTD_LINHAS_LIDAS;
    while(fimDoArq) {
        
        QNTD_LINHAS_LIDAS = geraMatriz();
       
        for (int i = 0; i < nomesArquivos.size(); ++i) {
            pairCodigoDescricao(nomesArquivos[i], i, QNTD_LINHAS_LIDAS);
        }
        
        cudaMalloc((void**)&d_categoricos, sizeof(dado) * LIN * COL_CAT_DATA);
        
        cudaMalloc((void**)&d_dicDados, sizeof(mapa) * COL_CAT_DATA * MAX_LIN_DICIONARIO);
        
        cudaMemcpy(d_categoricos, categoricos, sizeof(dado) * LIN * COL_CAT_DATA, cudaMemcpyHostToDevice);
        
        cudaMemcpy(d_dicDados, dicDados, sizeof(mapa) * COL_CAT_DATA * MAX_LIN_DICIONARIO, cudaMemcpyHostToDevice);
        
        int numBlock = getNumBlock(QNTD_LINHAS_LIDAS);
        int divisionTask = getDivisionTask(QNTD_LINHAS_LIDAS, cudaCore);
        /*
        cout << "NumBlock: " << numBlock << endl;
        cout << "DivisionTask: " << divisionTask << endl;
        cout << "TotalDeThreads: " << numBlock*128 << endl;
        */

        insercaoDeDados << <numBlock, 128>> > (d_categoricos, d_dicDados, divisionTask, QNTD_LINHAS_LIDAS, COL_CAT_DATA);

        cudaMemcpy(categoricos, d_categoricos, sizeof(mapa) * COL_CAT_DATA * LIN, cudaMemcpyDeviceToHost);

        //printf("%s %d\n", categoricos[0][0].d, categoricos[0][0].id);
        
        exportaMatriz(QNTD_LINHAS_LIDAS);
        cudaFree(d_categoricos);
        cudaFree(d_dicDados);
       // free(categoricos);
    }

    arquivoPrincipal.close();

    auto end = chrono::steady_clock::now();
    std::cout << "Tempo       : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;    // Ending of parallel region 
}

void exportaMatriz(int maxLinhas) {
    int colunaNumerica = 0;
    int colunaCategorica = 0;
    fstream saida;
    saida.open("saida.csv", fstream::app);

    for (int i = 0; i < maxLinhas; i++) {
        for (int j = 0; j < COL_CAT_DATA + COL_NUM_DATA; j++) {
            if (idxColuna.find(j) != idxColuna.end()) {
                saida << to_string(categoricos[i][colunaCategorica].id) << ',';
                //if(categoricos[i][colunaCategorica].id != 0) printf("%d\n", categoricos[i][colunaCategorica].id);
                colunaCategorica++;
            }
            else
            {
                saida << numericos[i][colunaNumerica].d << ',';
               // if(i == 0) printf("%d %s\n", j, numericos[i][colunaNumerica].d);
                colunaNumerica++;

            }
        }
        //cout << "FIM DA LINHA" << endl;
        colunaNumerica = 0;
        colunaCategorica = 0;
    }

    saida.close();
}
bool buscaMap(int indexDoArquivo, string dado) {
    bool naoVectorEstaVazio = !buscaRapidaDeDado[indexDoArquivo].empty();
    if (naoVectorEstaVazio) {
        if (buscaRapidaDeDado[indexDoArquivo].find(dado) != buscaRapidaDeDado[indexDoArquivo].end()) {
            return false;
        };
    }
    return true;
}

void pairCodigoDescricao(string nomeArquivo, int indexDaColuna, int numLinhasLidas) {
    std::fstream arquivo;
    
    arquivo.open(nomeArquivo, fstream::app);
    
    int linhaDicionario = buscaRapidaDeDado[indexDaColuna].size(); // algoritmo para saber a linha do dicionario
    
    for (int linha = 0;linha < numLinhasLidas; linha++) {
    
        string dado = categoricos[linha][indexDaColuna].d;

        if (buscaMap(indexDaColuna, dado)) {
            
            int ultimoIndex = buscaRapidaDeDado[indexDaColuna].size() + 1;
            
            buscaRapidaDeDado[indexDaColuna][dado] = ultimoIndex;
            
            arquivo << to_string(ultimoIndex) << "," << dado << endl;

            strncpy(dicDados[linhaDicionario][indexDaColuna].d, dado.c_str(), dado.size());
            
            dicDados[linhaDicionario][indexDaColuna].id = ultimoIndex;
            
            linhaDicionario++;
           
        }
        

    }
    
    arquivo.close();

}