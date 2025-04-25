#include "mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QImageReader>
#include <QImageWriter>
#include <QLabel>
#include <QSlider>
#include <QPushButton>
#include <QGroupBox>
#include <QTimer>
#include <QStatusBar>
#include <QDebug>
#include <QComboBox>
#include <QMessageBox>
#include <QInputDialog>
#include <QDialog>
#include <QFormLayout>

#define CUDA_CHECK(call) \
{ cudaError_t err = call; if (err != cudaSuccess) { \
            qCritical() << "CUDA error:" << cudaGetErrorString(err); \
            throw std::runtime_error("CUDA error"); \
    }}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent),
    input_image_(512, 512, QImage::Format_RGBA8888),
    output_image_(512, 512, QImage::Format_RGBA8888)
{
    input_image_.fill(Qt::black);
    output_image_.fill(Qt::black);
    createControls();
    setWindowTitle("Fisheye Dewarping Tool");
    resize(1200, 800);
    dewarper_ = std::make_unique<FisheyeDewarper>(
        input_image_.width(), input_image_.height(),
        output_image_.width(), output_image_.height()
        );
    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &MainWindow::updateDewarp);
}

void MainWindow::createControls() {
    QWidget* central = new QWidget;
    QVBoxLayout* mainLayout = new QVBoxLayout;

    QHBoxLayout* imageLayout = new QHBoxLayout;
    source_image_display_ = new QLabel;
    source_image_display_->setAlignment(Qt::AlignCenter);
    source_image_display_->setMinimumSize(320, 240);
    source_image_display_->setText("Source Image");
    imageLayout->addWidget(source_image_display_, 1);

    image_display_ = new QLabel;
    image_display_->setAlignment(Qt::AlignCenter);
    image_display_->setMinimumSize(320, 240);
    image_display_->setText("Dewarped Image");
    imageLayout->addWidget(image_display_, 1);

    mainLayout->addLayout(imageLayout, 1);

    QGroupBox* controlBox = new QGroupBox("Dewarping Parameters");
    QGridLayout* controlLayout = new QGridLayout;

    auto createSlider = [](float min, float max, float step) {
        QSlider* slider = new QSlider(Qt::Horizontal);
        slider->setRange(min / step, max / step);
        slider->setSingleStep(1);
        return slider;
    };

    pan_slider_ = createSlider(PAN_MIN, PAN_MAX, 0.01f);
    tilt_slider_ = createSlider(TILT_MIN, TILT_MAX, 0.01f);
    roll_slider_ = createSlider(ROLL_MIN, ROLL_MAX, 0.01f);
    zoom_slider_ = createSlider(ZOOM_MIN, ZOOM_MAX, 0.01f);
    hfov_slider_ = createSlider(FOV_MIN, FOV_MAX, 0.01f);
    vfov_slider_ = createSlider(FOV_MIN, FOV_MAX, 0.01f);
    output_width_slider_ = createSlider(OUTPUT_SIZE_MIN, OUTPUT_SIZE_MAX, 1.0f);
    output_height_slider_ = createSlider(OUTPUT_SIZE_MIN, OUTPUT_SIZE_MAX, 1.0f);

    zoom_slider_->setValue(1.0f / 0.01f);
    hfov_slider_->setValue(2.0f / 0.01f);
    vfov_slider_->setValue(2.0f / 0.01f);
    output_width_slider_->setValue(1024);
    output_height_slider_->setValue(1024);

    // Numeric labels
    pan_label_ = new QLabel("0.00");
    tilt_label_ = new QLabel("0.00");
    roll_label_ = new QLabel("0.00");
    zoom_label_ = new QLabel("1.00");
    hfov_label_ = new QLabel("2.00");
    vfov_label_ = new QLabel("2.00");
    output_width_label_ = new QLabel("1024");
    output_height_label_ = new QLabel("1024");

    // Projection model dropdown
    projection_model_combo_ = new QComboBox;
    projection_model_combo_->addItem("Equidistant", FisheyeDewarper::PROJECTION_EQUIDISTANT);
    projection_model_combo_->addItem("Equisolid", FisheyeDewarper::PROJECTION_EQUISOLID);
    projection_model_combo_->addItem("Stereographic", FisheyeDewarper::PROJECTION_STEREOGRAPHIC);
    projection_model_combo_->addItem("Orthographic", FisheyeDewarper::PROJECTION_ORTHOGRAPHIC);
    projection_model_combo_->addItem("Equirectangular", FisheyeDewarper::PROJECTION_EQUIRECTANGULAR);
    projection_model_combo_->addItem("Panini", FisheyeDewarper::PROJECTION_PANINI);
    projection_model_combo_->addItem("Cylindrical", FisheyeDewarper::PROJECTION_CYLINDRICAL);
    projection_model_combo_->setCurrentIndex(1);

    // View selection and management
    view_combo_ = new QComboBox;
    view_combo_->setEnabled(false);
    QPushButton* saveConfigBtn = new QPushButton("Save Config");
    QPushButton* loadConfigBtn = new QPushButton("Load Config");
    QPushButton* addViewBtn = new QPushButton("Add View");
    QPushButton* editViewBtn = new QPushButton("Edit View");
    QPushButton* deleteViewBtn = new QPushButton("Delete View");

    // Layout with sliders and labels
    int row = 0;
    controlLayout->addWidget(new QLabel("Pan"), row, 0);
    controlLayout->addWidget(pan_slider_, row, 1);
    controlLayout->addWidget(pan_label_, row, 2);
    row++;
    controlLayout->addWidget(new QLabel("Tilt"), row, 0);
    controlLayout->addWidget(tilt_slider_, row, 1);
    controlLayout->addWidget(tilt_label_, row, 2);
    row++;
    controlLayout->addWidget(new QLabel("Roll"), row, 0);
    controlLayout->addWidget(roll_slider_, row, 1);
    controlLayout->addWidget(roll_label_, row, 2);
    row++;
    controlLayout->addWidget(new QLabel("Zoom"), row, 0);
    controlLayout->addWidget(zoom_slider_, row, 1);
    controlLayout->addWidget(zoom_label_, row, 2);
    row++;
    controlLayout->addWidget(new QLabel("HFOV"), row, 0);
    controlLayout->addWidget(hfov_slider_, row, 1);
    controlLayout->addWidget(hfov_label_, row, 2);
    row++;
    controlLayout->addWidget(new QLabel("VFOV"), row, 0);
    controlLayout->addWidget(vfov_slider_, row, 1);
    controlLayout->addWidget(vfov_label_, row, 2);
    row++;
    controlLayout->addWidget(new QLabel("Output Width"), row, 0);
    controlLayout->addWidget(output_width_slider_, row, 1);
    controlLayout->addWidget(output_width_label_, row, 2);
    row++;
    controlLayout->addWidget(new QLabel("Output Height"), row, 0);
    controlLayout->addWidget(output_height_slider_, row, 1);
    controlLayout->addWidget(output_height_label_, row, 2);
    row++;
    controlLayout->addWidget(new QLabel("Projection Model"), row, 0);
    controlLayout->addWidget(projection_model_combo_, row, 1);
    row++;
    controlLayout->addWidget(new QLabel("View"), row, 0);
    controlLayout->addWidget(view_combo_, row, 1);
    row++;

    // Configuration buttons
    QHBoxLayout* configLayout = new QHBoxLayout;
    configLayout->addWidget(saveConfigBtn);
    configLayout->addWidget(loadConfigBtn);
    configLayout->addWidget(addViewBtn);
    configLayout->addWidget(editViewBtn);
    controlLayout->addLayout(configLayout, row, 0, 1, 2);
    row++;
    controlLayout->addWidget(deleteViewBtn, row, 0, 1, 2);

    // Connect configuration buttons
    connect(saveConfigBtn, &QPushButton::clicked, this, &MainWindow::saveConfig);
    connect(loadConfigBtn, &QPushButton::clicked, this, &MainWindow::loadConfig);
    connect(addViewBtn, &QPushButton::clicked, this, &MainWindow::addView);
    connect(editViewBtn, &QPushButton::clicked, this, &MainWindow::editView);
    connect(deleteViewBtn, &QPushButton::clicked, this, &MainWindow::deleteView);
    connect(view_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::applyView);

    connect(output_width_slider_, &QSlider::valueChanged, this, &MainWindow::updateOutputWidthLabel);
    connect(output_height_slider_, &QSlider::valueChanged, this, &MainWindow::updateOutputHeightLabel);
    connect(output_width_slider_, &QSlider::valueChanged, this, &MainWindow::updateOutputSize);
    connect(output_height_slider_, &QSlider::valueChanged, this, &MainWindow::updateOutputSize);

    connect(pan_slider_, &QSlider::valueChanged, this, &MainWindow::updatePanLabel);
    connect(tilt_slider_, &QSlider::valueChanged, this, &MainWindow::updateTiltLabel);
    connect(roll_slider_, &QSlider::valueChanged, this, &MainWindow::updateRollLabel);
    connect(zoom_slider_, &QSlider::valueChanged, this, &MainWindow::updateZoomLabel);
    connect(hfov_slider_, &QSlider::valueChanged, this, &MainWindow::updateHFOVLabel);
    connect(vfov_slider_, &QSlider::valueChanged, this, &MainWindow::updateVFOVLabel);

    QPushButton* loadBtn = new QPushButton("Load Image");
    QPushButton* saveBtn = new QPushButton("Save Result");
    connect(loadBtn, &QPushButton::clicked, this, &MainWindow::loadImage);
    connect(saveBtn, &QPushButton::clicked, this, &MainWindow::saveResult);

    QHBoxLayout* buttonLayout = new QHBoxLayout;
    buttonLayout->addWidget(loadBtn);
    buttonLayout->addWidget(saveBtn);

    controlBox->setLayout(controlLayout);
    mainLayout->addWidget(controlBox);
    mainLayout->addLayout(buttonLayout);

    central->setLayout(mainLayout);
    setCentralWidget(central);
}

void MainWindow::updatePanLabel(int value) {
    pan_label_->setText(QString::number(value * 0.01f, 'f', 2));
}

void MainWindow::updateTiltLabel(int value) {
    tilt_label_->setText(QString::number(value * 0.01f, 'f', 2));
}

void MainWindow::updateRollLabel(int value) {
    roll_label_->setText(QString::number(value * 0.01f, 'f', 2));
}

void MainWindow::updateZoomLabel(int value) {
    zoom_label_->setText(QString::number(value * 0.01f, 'f', 2));
}

void MainWindow::updateHFOVLabel(int value) {
    hfov_label_->setText(QString::number(value * 0.01f, 'f', 2));
}

void MainWindow::updateVFOVLabel(int value) {
    vfov_label_->setText(QString::number(value * 0.01f, 'f', 2));
}

void MainWindow::updateOutputWidthLabel(int value) {
    output_width_label_->setText(QString::number(value));
}

void MainWindow::updateOutputHeightLabel(int value) {
    output_height_label_->setText(QString::number(value));
}

void MainWindow::updateOutputSize() {
    int new_width = output_width_slider_->value();
    int new_height = output_height_slider_->value();

    output_image_ = QImage(new_width, new_height, QImage::Format_RGBA8888);
    output_image_.fill(Qt::black);

    dewarper_ = std::make_unique<FisheyeDewarper>(
        input_image_.width(), input_image_.height(),
        output_image_.width(), output_image_.height()
        );

    updateDewarp();
}

void MainWindow::updateDewarp() {
    if (input_image_.isNull()) return;

    uchar4* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, input_image_.width() * input_image_.height() * sizeof(uchar4)));
    CUDA_CHECK(cudaMemcpy(d_input, input_image_.constBits(),
                          input_image_.width() * input_image_.height() * sizeof(uchar4),
                          cudaMemcpyHostToDevice));

    int input_offset = (input_image_.height() / 4) * input_image_.width() + (input_image_.width() / 4);
    uchar4* input_pixel = (uchar4*)input_image_.constBits();
    qDebug() << "Input pixel (width/4, height/4): R=" << input_pixel[input_offset].x
             << "G=" << input_pixel[input_offset].y
             << "B=" << input_pixel[input_offset].z
             << "A=" << input_pixel[input_offset].w;

    uchar4* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, output_image_.width() * output_image_.height() * sizeof(uchar4)));

    FisheyeDewarper::DewarpParams params;
    params.pan = pan_slider_->value() * 0.01f;
    params.tilt = tilt_slider_->value() * 0.01f;
    params.roll = roll_slider_->value() * 0.01f;
    params.zoom = zoom_slider_->value() * 0.01f;
    params.hfov = hfov_slider_->value() * 0.01f;
    params.vfov = vfov_slider_->value() * 0.01f;
    params.model = static_cast<FisheyeDewarper::ProjectionModel>(projection_model_combo_->currentData().toInt());

    dewarper_->process(d_input, d_output, params);

    int output_offset = (output_image_.height() / 4) * output_image_.width() + (output_image_.width() / 4);
    unsigned char h_output_pixel[4];
    CUDA_CHECK(cudaMemcpy(h_output_pixel, d_output + output_offset,
                          sizeof(uchar4), cudaMemcpyDeviceToHost));
    qDebug() << "Output pixel (width/4, height/4) from GPU: R=" << (int)h_output_pixel[0]
             << "G=" << (int)h_output_pixel[1]
             << "B=" << (int)h_output_pixel[2]
             << "A=" << (int)h_output_pixel[3];

    CUDA_CHECK(cudaMemcpy(output_image_.bits(), d_output,
                          output_image_.width() * output_image_.height() * sizeof(uchar4),
                          cudaMemcpyDeviceToHost));

    uchar4* output_pixel = (uchar4*)output_image_.bits();
    qDebug() << "Output pixel (width/4, height/4) after copy: R=" << output_pixel[output_offset].x
             << "G=" << output_pixel[output_offset].y
             << "B=" << output_pixel[output_offset].z
             << "A=" << output_pixel[output_offset].w;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    updateDisplay();

    auto metrics = dewarper_->getMetrics();
    statusBar()->showMessage(QString("Frame: %1ms | Throughput: %2 MPixels/s")
                                 .arg(metrics.frame_time_ms, 0, 'f', 2)
                                 .arg(metrics.throughput_mpixels, 0, 'f', 1));
}

void MainWindow::updateDisplay() {
    QPixmap pixmap = QPixmap::fromImage(output_image_);
    image_display_->setPixmap(pixmap.scaled(
        image_display_->width(), image_display_->height(),
        Qt::KeepAspectRatio, Qt::SmoothTransformation
        ));
}

void MainWindow::updateSourceDisplay() {
    QPixmap pixmap = QPixmap::fromImage(input_image_);
    source_image_display_->setPixmap(pixmap.scaled(
        source_image_display_->width(), source_image_display_->height(),
        Qt::KeepAspectRatio, Qt::SmoothTransformation
        ));
}

void MainWindow::loadImage() {
    QString file = QFileDialog::getOpenFileName(this, "Open Image", "",
                                                "Images (*.png *.jpg *.bmp *.tif)");
    if (!file.isEmpty()) {
        QImageReader reader(file);
        input_image_ = reader.read().convertToFormat(QImage::Format_RGBA8888);
        output_image_ = QImage(input_image_.width(), input_image_.height(), QImage::Format_RGBA8888);
        output_image_.fill(Qt::black);
        qDebug() << "Input image format:" << input_image_.format();
        input_image_.save("debug_input.png");
        dewarper_ = std::make_unique<FisheyeDewarper>(
            input_image_.width(), input_image_.height(),
            output_image_.width(), output_image_.height()
            );
        output_width_slider_->setValue(input_image_.width());
        output_height_slider_->setValue(input_image_.height());
        updateOutputWidthLabel(input_image_.width());
        updateOutputHeightLabel(input_image_.height());
        updateSourceDisplay();
        timer_->start(33);
        updateDewarp();
    }
}

void MainWindow::saveResult() {
    QString file = QFileDialog::getSaveFileName(this, "Save Image", "",
                                                "PNG (*.png);;JPEG (*.jpg)");
    if (!file.isEmpty()) {
        QImageWriter writer(file);
        writer.write(output_image_);
    }
}

void MainWindow::saveConfig() {
    if (views_.empty()) {
        QMessageBox::warning(this, "Save Configuration", "No views to save.");
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(this, "Save Configuration", "",
                                                    "JSON Files (*.json)");
    if (fileName.isEmpty()) return;

    try {
        dewarper_->saveConfig(fileName.toStdString(), camera_name_, views_);
        current_config_file_ = fileName.toStdString();
        QMessageBox::information(this, "Save Configuration", "Configuration saved successfully.");
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Save Configuration", QString("Failed to save configuration: %1").arg(e.what()));
    }
}

void MainWindow::loadConfig() {
    QString fileName = QFileDialog::getOpenFileName(this, "Load Configuration", "",
                                                    "JSON Files (*.json)");
    if (fileName.isEmpty()) return;

    try {
        dewarper_->loadConfig(fileName.toStdString(), camera_name_, views_);
        view_combo_->clear();
        for (size_t i = 0; i < views_.size(); ++i) {
            view_combo_->addItem(QString::fromStdString(views_[i].name), static_cast<int>(i));
        }
        view_combo_->setEnabled(!views_.empty());
        if (!views_.empty()) {
            view_combo_->setCurrentIndex(0);
            applyView(0);
        }
        current_config_file_ = fileName.toStdString();
        QMessageBox::information(this, "Load Configuration", "Configuration loaded successfully.");
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Load Configuration", QString("Failed to load configuration: %1").arg(e.what()));
    }
}

void MainWindow::addView() {
    bool ok;
    QString name = QInputDialog::getText(this, "Add View", "View Name:", QLineEdit::Normal, "", &ok);
    if (!ok || name.isEmpty()) return;

    // Check for duplicate names
    for (const auto& view : views_) {
        if (view.name == name.toStdString()) {
            QMessageBox::warning(this, "Add View", "A view with this name already exists.");
            return;
        }
    }

    FisheyeDewarper::ViewConfig config;
    config.name = name.toStdString();
    config.pan = pan_slider_->value() * 0.01f;
    config.tilt = tilt_slider_->value() * 0.01f;
    config.roll = roll_slider_->value() * 0.01f;
    config.zoom = zoom_slider_->value() * 0.01f;
    config.hfov = hfov_slider_->value() * 0.01f;
    config.vfov = vfov_slider_->value() * 0.01f;
    config.output_width = output_width_slider_->value();
    config.output_height = output_height_slider_->value();
    config.model = static_cast<FisheyeDewarper::ProjectionModel>(projection_model_combo_->currentData().toInt());

    views_.push_back(config);
    view_combo_->addItem(name, static_cast<int>(views_.size() - 1));
    view_combo_->setEnabled(true);
    view_combo_->setCurrentIndex(static_cast<int>(views_.size() - 1));
}

void MainWindow::editView() {
    if (views_.empty()) {
        QMessageBox::warning(this, "Edit View", "No views to edit.");
        return;
    }

    int index = view_combo_->currentIndex();
    if (index < 0) return;

    auto& config = views_[index];
    bool ok;
    QString name = QInputDialog::getText(this, "Edit View", "View Name:", QLineEdit::Normal,
                                         QString::fromStdString(config.name), &ok);
    if (!ok || name.isEmpty()) return;

    // Check for duplicate names
    for (size_t i = 0; i < views_.size(); ++i) {
        if (i != static_cast<size_t>(index) && views_[i].name == name.toStdString()) {
            QMessageBox::warning(this, "Edit View", "A view with this name already exists.");
            return;
        }
    }

    config.name = name.toStdString();
    config.pan = pan_slider_->value() * 0.01f;
    config.tilt = tilt_slider_->value() * 0.01f;
    config.roll = roll_slider_->value() * 0.01f;
    config.zoom = zoom_slider_->value() * 0.01f;
    config.hfov = hfov_slider_->value() * 0.01f;
    config.vfov = vfov_slider_->value() * 0.01f;
    config.output_width = output_width_slider_->value();
    config.output_height = output_height_slider_->value();
    config.model = static_cast<FisheyeDewarper::ProjectionModel>(projection_model_combo_->currentData().toInt());

    view_combo_->setItemText(index, name);
}

void MainWindow::deleteView() {
    if (views_.empty()) {
        QMessageBox::warning(this, "Delete View", "No views to delete.");
        return;
    }

    int index = view_combo_->currentIndex();
    if (index < 0) return;

    views_.erase(views_.begin() + index);
    view_combo_->removeItem(index);

    if (views_.empty()) {
        view_combo_->setEnabled(false);
    } else {
        view_combo_->setCurrentIndex(0);
        applyView(0);
    }
}

void MainWindow::applyView(int index) {
    if (index < 0 || static_cast<size_t>(index) >= views_.size()) return;

    const auto& config = views_[index];
    pan_slider_->setValue(static_cast<int>(config.pan / 0.01f));
    tilt_slider_->setValue(static_cast<int>(config.tilt / 0.01f));
    roll_slider_->setValue(static_cast<int>(config.roll / 0.01f));
    zoom_slider_->setValue(static_cast<int>(config.zoom / 0.01f));
    hfov_slider_->setValue(static_cast<int>(config.hfov / 0.01f));
    vfov_slider_->setValue(static_cast<int>(config.vfov / 0.01f));
    output_width_slider_->setValue(config.output_width);
    output_height_slider_->setValue(config.output_height);
    projection_model_combo_->setCurrentIndex(projection_model_combo_->findData(static_cast<int>(config.model)));

    updatePanLabel(pan_slider_->value());
    updateTiltLabel(tilt_slider_->value());
    updateRollLabel(roll_slider_->value());
    updateZoomLabel(zoom_slider_->value());
    updateHFOVLabel(hfov_slider_->value());
    updateVFOVLabel(vfov_slider_->value());
    updateOutputWidthLabel(output_width_slider_->value());
    updateOutputHeightLabel(output_height_slider_->value());
}
