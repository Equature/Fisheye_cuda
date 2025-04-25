#pragma once

#include <QMainWindow>
#include <QComboBox>
#include <QLabel>
#include <QList>
#include <memory>
#include "fisheye_dewarper.h"

// Slider limits
#define PAN_MIN -M_PI
#define PAN_MAX M_PI
#define TILT_MIN -M_PI / 2.0f
#define TILT_MAX M_PI / 2.0f
#define ROLL_MIN -M_PI
#define ROLL_MAX M_PI
#define ZOOM_MIN 0.1f
#define ZOOM_MAX 4.0f
#define FOV_MIN 0.5f
#define FOV_MAX 3.0f
#define OUTPUT_SIZE_MIN 128
#define OUTPUT_SIZE_MAX 2048

class QSlider;
class QStatusBar;
class QTimer;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);

private slots:
    void updateDewarp();
    void loadImage();
    void saveResult();
    void updateOutputSize();
    void updatePanLabel(int value);
    void updateTiltLabel(int value);
    void updateRollLabel(int value);
    void updateZoomLabel(int value);
    void updateHFOVLabel(int value);
    void updateVFOVLabel(int value);
    void updateOutputWidthLabel(int value);
    void updateOutputHeightLabel(int value);
    void saveConfig();
    void loadConfig();
    void addView();
    void editView();
    void deleteView();
    void applyView(int index);

private:
    void createControls();
    void updateDisplay();
    void updateSourceDisplay();

    std::unique_ptr<FisheyeDewarper> dewarper_;
    QImage input_image_;
    QImage output_image_;
    QTimer* timer_;

    // UI Controls
    QLabel* image_display_;
    QLabel* source_image_display_;
    QSlider* pan_slider_;
    QSlider* tilt_slider_;
    QSlider* roll_slider_;
    QSlider* zoom_slider_;
    QSlider* hfov_slider_;
    QSlider* vfov_slider_;
    QSlider* output_width_slider_;
    QSlider* output_height_slider_;
    QComboBox* projection_model_combo_;
    // Numeric labels for sliders
    QLabel* pan_label_;
    QLabel* tilt_label_;
    QLabel* roll_label_;
    QLabel* zoom_label_;
    QLabel* hfov_label_;
    QLabel* vfov_label_;
    QLabel* output_width_label_;
    QLabel* output_height_label_;
    // Configuration management
    QComboBox* view_combo_;
    std::vector<FisheyeDewarper::ViewConfig> views_;
    std::string camera_name_;
    std::string current_config_file_;
};
