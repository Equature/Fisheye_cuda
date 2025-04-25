#include "mainwindow.h"
#include <QApplication>
#include <fstream>
#include <nlohmann/json.hpp>

int main(int argc, char *argv[]) {

    std::ifstream f("/root/shared/dewarp/camera1.json");
    nlohmann::json data = nlohmann::json::parse(f);


    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
