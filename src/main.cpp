#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>
#include <TJpg_Decoder.h>
#include <esp_heap_caps.h>

#include "TensorFlowLite_ESP32.h"
#include "scar_fomo_int8_model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"


// ============================================================
// WIFI ACCESS POINT
// ============================================================

const char* AP_SSID = "Railway_FOMO_ESP32";
const char* AP_PASSWORD = "railway123";

WebServer server(80);


// ============================================================
// TENSORFLOW LITE
// ============================================================

tflite::AllOpsResolver resolver;

tflite::MicroErrorReporter micro_error_reporter;

const tflite::Model* model = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


// ============================================================
// TENSOR ARENA
// ============================================================

constexpr size_t TENSOR_ARENA_SIZE = 96 * 1024;

uint8_t* tensor_arena = nullptr;


// ============================================================
// IMAGE SETTINGS
// ============================================================

constexpr int IMAGE_SIZE = 96;

constexpr int IMAGE_PIXELS =
    IMAGE_SIZE * IMAGE_SIZE;

constexpr int IMAGE_BUFFER_SIZE =
    IMAGE_PIXELS * 3;


// NCHW INT8 image buffer
int8_t image_buffer[IMAGE_BUFFER_SIZE];


// ============================================================
// FOMO GRID
// ============================================================

constexpr int FOMO_GRID_SIZE = 12;

constexpr int FOMO_GRID_CELLS =
    FOMO_GRID_SIZE * FOMO_GRID_SIZE;


// Store all 144 FOMO probabilities
float fomo_grid[FOMO_GRID_CELLS];


// ============================================================
// SPIFFS JPEG STORAGE
// ============================================================

const char* JPEG_PATH =
    "/railway_upload.jpg";

File uploadFile;

size_t jpeg_size = 0;

bool upload_ok = false;


// Maximum JPEG upload size
constexpr size_t MAX_JPEG_SIZE =
    512 * 1024;


// ============================================================
// JPEG DIMENSIONS
// ============================================================

int original_width = 0;

int original_height = 0;

int decoded_width = 0;

int decoded_height = 0;


// ============================================================
// DETECTION THRESHOLD
// ============================================================

constexpr float SCAR_THRESHOLD = 0.90f;


// ============================================================
// RGB888 -> INT8
// ============================================================

inline int8_t pixelToInt8(uint8_t value)
{
    return (int8_t)((int)value - 128);
}


// ============================================================
// JPEG CALLBACK
// ============================================================
//
// JPEG:
// RGB565
//   ↓
// RGB888
//   ↓
// 96 x 96
//   ↓
// NCHW INT8
//
// ============================================================

bool jpegCallback(
    int16_t x,
    int16_t y,
    uint16_t w,
    uint16_t h,
    uint16_t* bitmap
)
{
    if (
        decoded_width <= 0 ||
        decoded_height <= 0
    )
    {
        return false;
    }


    for (
        uint16_t row = 0;
        row < h;
        row++
    )
    {
        for (
            uint16_t col = 0;
            col < w;
            col++
        )
        {
            int src_x =
                x + col;

            int src_y =
                y + row;


            if (
                src_x < 0 ||
                src_y < 0 ||
                src_x >= decoded_width ||
                src_y >= decoded_height
            )
            {
                continue;
            }


            uint16_t pixel =
                bitmap[
                    row * w + col
                ];


            // ------------------------------------------------
            // RGB565 -> RGB888
            // ------------------------------------------------

            uint8_t r =
                ((pixel >> 11) & 0x1F)
                * 255 / 31;


            uint8_t g =
                ((pixel >> 5) & 0x3F)
                * 255 / 63;


            uint8_t b =
                (pixel & 0x1F)
                * 255 / 31;


            // ------------------------------------------------
            // Map decoded image to 96 x 96
            // ------------------------------------------------

            int dst_x =
                (src_x * IMAGE_SIZE)
                / decoded_width;


            int dst_y =
                (src_y * IMAGE_SIZE)
                / decoded_height;


            if (
                dst_x < 0 ||
                dst_x >= IMAGE_SIZE ||
                dst_y < 0 ||
                dst_y >= IMAGE_SIZE
            )
            {
                continue;
            }


            int index =
                dst_y * IMAGE_SIZE +
                dst_x;


            // ------------------------------------------------
            // NCHW
            // ------------------------------------------------

            image_buffer[index] =
                pixelToInt8(r);


            image_buffer[
                IMAGE_PIXELS + index
            ] =
                pixelToInt8(g);


            image_buffer[
                (2 * IMAGE_PIXELS) + index
            ] =
                pixelToInt8(b);
        }
    }


    return true;
}


// ============================================================
// INITIALIZE MODEL
// ============================================================

bool initializeModel()
{
    Serial.println();

    Serial.println(
        "================================"
    );

    Serial.println(
        "Initializing Scar FOMO model..."
    );

    Serial.println(
        "================================"
    );


    // --------------------------------------------------------
    // Load model
    // --------------------------------------------------------

    model =
        tflite::GetModel(
            scar_fomo_int8_model
        );


    if (model == nullptr)
    {
        Serial.println(
            "ERROR: Model is NULL."
        );

        return false;
    }


    Serial.printf(
        "Model schema version: %d\n",
        model->version()
    );


    if (
        model->version() !=
        TFLITE_SCHEMA_VERSION
    )
    {
        Serial.println(
            "ERROR: Model schema mismatch."
        );

        return false;
    }


    // --------------------------------------------------------
    // Allocate tensor arena
    // --------------------------------------------------------

    tensor_arena =
        (uint8_t*)heap_caps_malloc(
            TENSOR_ARENA_SIZE,
            MALLOC_CAP_8BIT
        );


    if (tensor_arena == nullptr)
    {
        Serial.println(
            "ERROR: Tensor arena allocation failed."
        );

        return false;
    }


    Serial.printf(
        "Tensor arena allocated: %u bytes\n",
        (unsigned int)
        TENSOR_ARENA_SIZE
    );


    Serial.printf(
        "Free heap after arena allocation: %u bytes\n",
        ESP.getFreeHeap()
    );


    // --------------------------------------------------------
    // Create interpreter
    // --------------------------------------------------------

    interpreter =
        new tflite::MicroInterpreter(
            model,
            resolver,
            tensor_arena,
            TENSOR_ARENA_SIZE,
            &micro_error_reporter
        );


    if (interpreter == nullptr)
    {
        Serial.println(
            "ERROR: Interpreter creation failed."
        );

        return false;
    }


    Serial.println(
        "Interpreter created."
    );


    // --------------------------------------------------------
    // Allocate tensors
    // --------------------------------------------------------

    TfLiteStatus status =
        interpreter->AllocateTensors();


    if (status != kTfLiteOk)
    {
        Serial.println(
            "ERROR: AllocateTensors failed."
        );

        return false;
    }


    Serial.println(
        "Tensors allocated successfully."
    );


    // --------------------------------------------------------
    // Get tensors
    // --------------------------------------------------------

    input =
        interpreter->input(0);

    output =
        interpreter->output(0);


    if (
        input == nullptr ||
        output == nullptr
    )
    {
        Serial.println(
            "ERROR: Input/output tensor unavailable."
        );

        return false;
    }


    // ========================================================
    // INPUT INFORMATION
    // ========================================================

    Serial.println();

    Serial.println(
        "========== MODEL INPUT =========="
    );


    Serial.printf(
        "Type: %d\n",
        input->type
    );


    Serial.printf(
        "Shape: %d x %d x %d x %d\n",
        input->dims->data[0],
        input->dims->data[1],
        input->dims->data[2],
        input->dims->data[3]
    );


    Serial.printf(
        "Scale: %.8f\n",
        input->params.scale
    );


    Serial.printf(
        "Zero point: %d\n",
        input->params.zero_point
    );


    // ========================================================
    // OUTPUT INFORMATION
    // ========================================================

    Serial.println();

    Serial.println(
        "========== MODEL OUTPUT ========="
    );


    Serial.printf(
        "Type: %d\n",
        output->type
    );


    Serial.printf(
        "Shape: %d x %d x %d x %d\n",
        output->dims->data[0],
        output->dims->data[1],
        output->dims->data[2],
        output->dims->data[3]
    );


    Serial.printf(
        "Scale: %.8f\n",
        output->params.scale
    );


    Serial.printf(
        "Zero point: %d\n",
        output->params.zero_point
    );


    Serial.println(
        "================================"
    );


    Serial.println(
        "Scar FOMO model initialized."
    );


    return true;
}


// ============================================================
// RUN FOMO INFERENCE
// ============================================================

bool runInference(
    float& max_probability,
    int& best_x,
    int& best_y,
    unsigned long& inference_time
)
{
    if (
        interpreter == nullptr ||
        input == nullptr ||
        output == nullptr
    )
    {
        return false;
    }


    // --------------------------------------------------------
    // Copy NCHW INT8 image into model input
    // --------------------------------------------------------

    memcpy(
        input->data.int8,
        image_buffer,
        IMAGE_BUFFER_SIZE
    );


    // --------------------------------------------------------
    // Run inference
    // --------------------------------------------------------

    unsigned long start_time =
        millis();


    TfLiteStatus status =
        interpreter->Invoke();


    inference_time =
        millis() - start_time;


    if (status != kTfLiteOk)
    {
        Serial.println(
            "ERROR: Model inference failed."
        );

        return false;
    }


    // --------------------------------------------------------
    // Reset strongest cell
    // --------------------------------------------------------

    max_probability = 0.0f;

    best_x = 0;

    best_y = 0;


    // --------------------------------------------------------
    // Read all 144 FOMO cells
    // --------------------------------------------------------

    for (
        int y = 0;
        y < FOMO_GRID_SIZE;
        y++
    )
    {
        for (
            int x = 0;
            x < FOMO_GRID_SIZE;
            x++
        )
        {
            int index =
                y * FOMO_GRID_SIZE + x;


            int8_t quantized =
                output->data.int8[index];


            // ------------------------------------------------
            // INT8 -> probability
            // ------------------------------------------------

            float probability =
                (
                    quantized -
                    output->params.zero_point
                )
                *
                output->params.scale;


            // Clamp

            if (
                probability < 0.0f
            )
            {
                probability = 0.0f;
            }


            if (
                probability > 1.0f
            )
            {
                probability = 1.0f;
            }


            // ------------------------------------------------
            // Save FOMO probability
            // ------------------------------------------------

            fomo_grid[index] =
                probability;


            // ------------------------------------------------
            // Find strongest cell
            // ------------------------------------------------

            if (
                probability >
                max_probability
            )
            {
                max_probability =
                    probability;

                best_x = x;

                best_y = y;
            }
        }
    }


    return true;
}


// ============================================================
// DASHBOARD HTML
// ============================================================

const char DASHBOARD_HTML[] PROGMEM = R"rawliteral(

<!DOCTYPE html>

<html>

<head>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<title>
Railway FOMO ESP32
</title>


<style>

body
{
    margin: 0;

    padding: 25px;

    font-family:
        Arial,
        sans-serif;

    background: #f4f4f4;
}


.container
{
    max-width: 900px;

    margin: auto;

    background: white;

    padding: 30px;

    border-radius: 18px;

    box-shadow:
        0 5px 25px
        rgba(0,0,0,0.12);
}


h1
{
    text-align: center;

    margin-bottom: 5px;
}


.subtitle
{
    text-align: center;

    color: #555;

    margin-bottom: 30px;
}


.status-online
{
    background: #e8f5e9;

    padding: 12px;

    border-radius: 10px;

    text-align: center;

    margin-bottom: 20px;

    font-weight: bold;
}


.upload-box
{
    border: 2px dashed #888;

    padding: 25px;

    text-align: center;

    border-radius: 15px;

    margin-bottom: 20px;
}


input
{
    max-width: 100%;
}


button
{
    padding: 13px 25px;

    border: none;

    border-radius: 8px;

    background: #222;

    color: white;

    font-size: 16px;

    cursor: pointer;
}


button:disabled
{
    background: #aaa;

    cursor: wait;
}


.preview
{
    text-align: center;

    margin-top: 20px;
}


.image-container
{
    position: relative;

    display: inline-block;

    max-width: 100%;
}


.image-container img
{
    display: block;

    max-width: 100%;

    max-height: 500px;

    border-radius: 10px;
}


#fomoCanvas
{
    position: absolute;

    left: 0;

    top: 0;

    width: 100%;

    height: 100%;

    pointer-events: none;

    border-radius: 10px;
}


.fomo-title
{
    margin-top: 20px;

    font-size: 20px;

    font-weight: bold;
}


.fomo-legend
{
    margin-top: 10px;

    font-size: 14px;

    color: #555;
}


.result
{
    margin-top: 25px;

    padding: 25px;

    border-radius: 15px;

    text-align: center;

    display: none;
}


.normal
{
    background: #e8f5e9;
}


.scar
{
    background: #ffebee;
}


.result-title
{
    font-size: 30px;

    font-weight: bold;

    margin-bottom: 15px;
}


.info
{
    font-size: 17px;

    line-height: 1.8;
}


.status
{
    text-align: center;

    margin-top: 15px;

    color: #666;
}


.small
{
    margin-top: 12px;

    font-size: 13px;

    color: #777;
}


.grid-info
{
    margin-top: 20px;

    padding: 15px;

    border-radius: 10px;

    background: #f5f5f5;

    text-align: center;

    font-size: 15px;
}

</style>

</head>


<body>


<div class="container">


<h1>
🚆 Railway FOMO ESP32
</h1>


<div class="subtitle">
Edge AI Railway Track Scar Detection
</div>


<div class="status-online">
🟢 ESP32 EDGE AI SYSTEM ONLINE
</div>


<div class="upload-box">


<h2>
Upload Railway Image
</h2>


<input
    type="file"
    id="imageInput"
    accept=".jpg,.jpeg,image/jpeg">


<br>
<br>


<button
    id="analyzeButton"
    onclick="analyzeImage()">

ANALYZE IMAGE

</button>


<div class="small">
JPEG image • Maximum 512 KB
</div>


</div>


<!-- ========================================================
     IMAGE + FOMO GRID
     ======================================================== -->

<div
    class="preview"
    id="preview">


<div
    class="image-container"
    id="imageContainer">


<img
    id="previewImage"
    style="display:none;">


<canvas
    id="fomoCanvas"
    style="display:none;">
</canvas>


</div>


</div>


<div
    class="fomo-title"
    id="fomoTitle"
    style="display:none;text-align:center;">

FOMO 12 × 12 Detection Grid

</div>


<div
    class="fomo-legend"
    id="fomoLegend"
    style="display:none;text-align:center;">

Transparent red areas = FOMO probability<br>

Yellow box = strongest FOMO cell

</div>


<div
    class="status"
    id="status">

ESP32 Ready

</div>


<!-- ========================================================
     RESULT
     ======================================================== -->

<div
    class="result"
    id="result">


<div
    class="result-title"
    id="resultTitle">
</div>


<div
    class="info"
    id="resultInfo">
</div>


<div
    class="grid-info"
    id="gridInfo">

</div>


</div>


</div>


<script>


// ============================================================
// ELEMENTS
// ============================================================

const imageInput =
    document.getElementById(
        "imageInput"
    );


const previewImage =
    document.getElementById(
        "previewImage"
    );


const fomoCanvas =
    document.getElementById(
        "fomoCanvas"
    );


const statusText =
    document.getElementById(
        "status"
    );


const resultBox =
    document.getElementById(
        "result"
    );


const resultTitle =
    document.getElementById(
        "resultTitle"
    );


const resultInfo =
    document.getElementById(
        "resultInfo"
    );


const gridInfo =
    document.getElementById(
        "gridInfo"
    );


const fomoTitle =
    document.getElementById(
        "fomoTitle"
    );


const fomoLegend =
    document.getElementById(
        "fomoLegend"
    );


const analyzeButton =
    document.getElementById(
        "analyzeButton"
    );


// ============================================================
// IMAGE SELECTION
// ============================================================

imageInput.addEventListener(
    "change",
    function()
    {
        const file =
            imageInput.files[0];


        if (!file)
        {
            return;
        }


        // ----------------------------------------------------
        // JPEG check
        // ----------------------------------------------------

        const isJpeg =
            file.type.includes("jpeg") ||
            file.name
                .toLowerCase()
                .endsWith(".jpg") ||
            file.name
                .toLowerCase()
                .endsWith(".jpeg");


        if (!isJpeg)
        {
            alert(
                "Please select a JPEG image."
            );

            imageInput.value = "";

            return;
        }


        // ----------------------------------------------------
        // Size check
        // ----------------------------------------------------

        if (
            file.size >
            512 * 1024
        )
        {
            alert(
                "JPEG is larger than 512 KB."
            );

            imageInput.value = "";

            return;
        }


        // ----------------------------------------------------
        // Browser preview
        // ----------------------------------------------------

        const imageURL =
            URL.createObjectURL(
                file
            );


        previewImage.src =
            imageURL;


        previewImage.style.display =
            "block";


        // Hide old FOMO grid

        fomoCanvas.style.display =
            "none";


        fomoTitle.style.display =
            "none";


        fomoLegend.style.display =
            "none";


        resultBox.style.display =
            "none";


        statusText.innerText =
            "Image selected. Ready for ESP32 analysis.";
    }
);


// ============================================================
// DRAW FOMO GRID
// ============================================================

function drawFomoGrid(
    probabilities,
    bestX,
    bestY
)
{
    if (
        !probabilities ||
        probabilities.length !== 144
    )
    {
        console.error(
            "Invalid FOMO grid."
        );

        return;
    }


    // --------------------------------------------------------
    // Get displayed image size
    // --------------------------------------------------------

    const width =
        previewImage.clientWidth;


    const height =
        previewImage.clientHeight;


    if (
        width <= 0 ||
        height <= 0
    )
    {
        return;
    }


    // --------------------------------------------------------
    // Canvas size
    // --------------------------------------------------------

    fomoCanvas.width =
        width;


    fomoCanvas.height =
        height;


    fomoCanvas.style.display =
        "block";


    fomoTitle.style.display =
        "block";


    fomoLegend.style.display =
        "block";


    const ctx =
        fomoCanvas.getContext(
            "2d"
        );


    const cellWidth =
        width / 12;


    const cellHeight =
        height / 12;


    // ========================================================
    // HEATMAP
    // ========================================================

    for (
        let y = 0;
        y < 12;
        y++
    )
    {
        for (
            let x = 0;
            x < 12;
            x++
        )
        {
            const index =
                y * 12 + x;


            const probability =
                probabilities[index];


            // Ignore very low probabilities

            if (
                probability < 0.03
            )
            {
                continue;
            }


            // ------------------------------------------------
            // Probability controls opacity
            // ------------------------------------------------

            const alpha =
                Math.min(
                    0.70,
                    probability * 0.70
                );


            ctx.fillStyle =
                "rgba(255, 0, 0, " +
                alpha +
                ")";


            ctx.fillRect(
                x * cellWidth,
                y * cellHeight,
                cellWidth,
                cellHeight
            );
        }
    }


    // ========================================================
    // GRID LINES
    // ========================================================

    ctx.lineWidth = 1;

    ctx.strokeStyle =
        "rgba(255,255,255,0.65)";


    for (
        let i = 0;
        i <= 12;
        i++
    )
    {
        // Vertical

        ctx.beginPath();

        ctx.moveTo(
            i * cellWidth,
            0
        );

        ctx.lineTo(
            i * cellWidth,
            height
        );

        ctx.stroke();


        // Horizontal

        ctx.beginPath();

        ctx.moveTo(
            0,
            i * cellHeight
        );

        ctx.lineTo(
            width,
            i * cellHeight
        );

        ctx.stroke();
    }


    // ========================================================
    // STRONGEST CELL
    // ========================================================

    ctx.lineWidth = 4;

    ctx.strokeStyle =
        "yellow";


    ctx.strokeRect(
        bestX * cellWidth + 2,
        bestY * cellHeight + 2,
        cellWidth - 4,
        cellHeight - 4
    );


    // ========================================================
    // STRONGEST CELL CENTER
    // ========================================================

    ctx.fillStyle =
        "yellow";


    ctx.beginPath();


    ctx.arc(
        bestX * cellWidth +
            cellWidth / 2,

        bestY * cellHeight +
            cellHeight / 2,

        Math.min(
            cellWidth,
            cellHeight
        ) * 0.12,

        0,

        Math.PI * 2
    );


    ctx.fill();


    // ========================================================
    // CONFIDENCE TEXT
    // ========================================================

    const strongestProbability =
        probabilities[
            bestY * 12 + bestX
        ];


    ctx.fillStyle =
        "white";


    ctx.font =
        "bold 12px Arial";


    ctx.textAlign =
        "center";


    ctx.textBaseline =
        "middle";


    // Text shadow

    ctx.shadowColor =
        "black";


    ctx.shadowBlur =
        3;


    ctx.fillText(
        (
            strongestProbability *
            100
        ).toFixed(0) + "%",

        bestX * cellWidth +
            cellWidth / 2,

        bestY * cellHeight +
            cellHeight / 2
    );


    ctx.shadowBlur =
        0;
}


// ============================================================
// ANALYZE IMAGE
// ============================================================

async function analyzeImage()
{
    const file =
        imageInput.files[0];


    if (!file)
    {
        alert(
            "Please select a JPEG image first."
        );

        return;
    }


    // --------------------------------------------------------
    // Size check
    // --------------------------------------------------------

    if (
        file.size >
        512 * 1024
    )
    {
        alert(
            "JPEG is larger than 512 KB."
        );

        return;
    }


    // --------------------------------------------------------
    // Disable button
    // --------------------------------------------------------

    analyzeButton.disabled =
        true;


    resultBox.style.display =
        "none";


    fomoCanvas.style.display =
        "none";


    fomoTitle.style.display =
        "none";


    fomoLegend.style.display =
        "none";


    statusText.innerText =
        "Uploading image to ESP32...";


    // --------------------------------------------------------
    // Form data
    // --------------------------------------------------------

    const formData =
        new FormData();


    formData.append(
        "image",
        file,
        file.name
    );


    try
    {
        statusText.innerText =
            "ESP32 is decoding the JPEG and running AI...";


        // ----------------------------------------------------
        // Send image to ESP32
        // ----------------------------------------------------

        const response =
            await fetch(
                "/upload",
                {
                    method: "POST",

                    body: formData
                }
            );


        // ----------------------------------------------------
        // Get JSON
        // ----------------------------------------------------

        const data =
            await response.json();


        if (!response.ok)
        {
            throw new Error(
                data.error ||
                "ESP32 processing failed."
            );
        }


        // ====================================================
        // DRAW FOMO GRID
        // ====================================================

        drawFomoGrid(
            data.fomo_grid,
            data.cell_x,
            data.cell_y
        );


        // ====================================================
        // SHOW RESULT
        // ====================================================

        resultBox.style.display =
            "block";


        // ====================================================
        // SCAR DETECTED
        // ====================================================

        if (data.detected)
        {
            resultBox.className =
                "result scar";


            resultTitle.innerText =
                "🔴 SCAR DETECTED";


            resultInfo.innerHTML =
                "<b>TRACK ABNORMAL</b>" +

                "<br><br>" +

                "Confidence: " +

                (
                    data.confidence *
                    100
                ).toFixed(2) +

                "%" +

                "<br>" +

                "Strongest Cell: (" +

                data.cell_x +

                ", " +

                data.cell_y +

                ")" +

                "<br>" +

                "Inference Time: " +

                data.inference_ms +

                " ms" +

                "<br>" +

                "JPEG Decode Time: " +

                data.decode_ms +

                " ms" +

                "<br>" +

                "Image: " +

                data.image_width +

                " × " +

                data.image_height;
        }


        // ====================================================
        // NORMAL TRACK
        // ====================================================

        else
        {
            resultBox.className =
                "result normal";


            resultTitle.innerText =
                "🟢 NORMAL TRACK";


            resultInfo.innerHTML =
                "<b>NO SCAR DETECTED</b>" +

                "<br><br>" +

                "Confidence: " +

                (
                    data.confidence *
                    100
                ).toFixed(2) +

                "%" +

                "<br>" +

                "Strongest Cell: (" +

                data.cell_x +

                ", " +

                data.cell_y +

                ")" +

                "<br>" +

                "Inference Time: " +

                data.inference_ms +

                " ms" +

                "<br>" +

                "JPEG Decode Time: " +

                data.decode_ms +

                " ms" +

                "<br>" +

                "Image: " +

                data.image_width +

                " × " +

                data.image_height;
        }


        // ====================================================
        // GRID INFORMATION
        // ====================================================

        gridInfo.innerHTML =
            "FOMO Output: <b>12 × 12 = 144 cells</b>" +

            "<br>" +

            "The heatmap shows the spatial probability " +

            "generated by the Scar FOMO model on the ESP32.";


        statusText.innerText =
            "Analysis complete — AI inference performed on ESP32.";
    }


    catch(error)
    {
        console.error(error);


        statusText.innerText =
            "Error: " +
            error.message;


        resultBox.style.display =
            "none";
    }


    // --------------------------------------------------------
    // Enable button
    // --------------------------------------------------------

    analyzeButton.disabled =
        false;
}

</script>


</body>

</html>

)rawliteral";


// ============================================================
// HANDLE JPEG UPLOAD
// ============================================================

void handleUploadData()
{
    HTTPUpload& upload =
        server.upload();


    switch (upload.status)
    {
        // ====================================================
        // START
        // ====================================================

        case UPLOAD_FILE_START:
        {
            Serial.println();

            Serial.println(
                "================================"
            );

            Serial.println(
                "New JPEG upload"
            );

            Serial.println(
                "================================"
            );


            Serial.printf(
                "Filename: %s\n",
                upload.filename.c_str()
            );


            Serial.printf(
                "Free heap: %u bytes\n",
                ESP.getFreeHeap()
            );


            jpeg_size = 0;

            upload_ok = false;


            // Remove previous JPEG

            if (
                SPIFFS.exists(
                    JPEG_PATH
                )
            )
            {
                SPIFFS.remove(
                    JPEG_PATH
                );
            }


            // Open new JPEG file

            uploadFile =
                SPIFFS.open(
                    JPEG_PATH,
                    FILE_WRITE
                );


            if (!uploadFile)
            {
                Serial.println(
                    "ERROR: Could not open JPEG file."
                );

                return;
            }


            Serial.println(
                "JPEG file opened in SPIFFS."
            );


            break;
        }


        // ====================================================
        // WRITE
        // ====================================================

        case UPLOAD_FILE_WRITE:
        {
            if (!uploadFile)
            {
                Serial.println(
                    "ERROR: JPEG file is not open."
                );

                return;
            }


            // ------------------------------------------------
            // Maximum JPEG size
            // ------------------------------------------------

            if (
                jpeg_size +
                upload.currentSize >
                MAX_JPEG_SIZE
            )
            {
                Serial.println(
                    "ERROR: JPEG exceeds 512 KB."
                );


                uploadFile.close();


                SPIFFS.remove(
                    JPEG_PATH
                );


                jpeg_size = 0;

                upload_ok = false;


                return;
            }


            // ------------------------------------------------
            // Write directly to SPIFFS
            // ------------------------------------------------

            size_t written =
                uploadFile.write(
                    upload.buf,
                    upload.currentSize
                );


            if (
                written !=
                upload.currentSize
            )
            {
                Serial.println(
                    "ERROR: Failed writing JPEG to SPIFFS."
                );


                uploadFile.close();


                SPIFFS.remove(
                    JPEG_PATH
                );


                jpeg_size = 0;

                upload_ok = false;


                return;
            }


            jpeg_size +=
                upload.currentSize;


            break;
        }


        // ====================================================
        // END
        // ====================================================

        case UPLOAD_FILE_END:
        {
            if (uploadFile)
            {
                uploadFile.close();
            }


            upload_ok = true;


            Serial.printf(
                "Upload complete: %u bytes\n",
                (unsigned int)jpeg_size
            );


            Serial.printf(
                "Free heap after upload: %u bytes\n",
                ESP.getFreeHeap()
            );


            Serial.println(
                "JPEG stored in SPIFFS."
            );


            break;
        }


        // ====================================================
        // ABORTED
        // ====================================================

        case UPLOAD_FILE_ABORTED:
        {
            Serial.println(
                "ERROR: Upload aborted."
            );


            if (uploadFile)
            {
                uploadFile.close();
            }


            SPIFFS.remove(
                JPEG_PATH
            );


            jpeg_size = 0;

            upload_ok = false;


            break;
        }
    }
}


// ============================================================
// PROCESS UPLOADED JPEG
// ============================================================

void handleUploadComplete()
{
    // --------------------------------------------------------
    // Validate upload
    // --------------------------------------------------------

    if (!upload_ok)
    {
        Serial.println(
            "ERROR: JPEG upload failed."
        );


        server.send(
            500,
            "application/json",
            "{\"error\":\"JPEG upload failed\"}"
        );


        return;
    }


    if (
        !SPIFFS.exists(
            JPEG_PATH
        )
    )
    {
        Serial.println(
            "ERROR: JPEG file not found."
        );


        server.send(
            500,
            "application/json",
            "{\"error\":\"Uploaded JPEG file not found\"}"
        );


        return;
    }


    if (
        jpeg_size == 0
    )
    {
        Serial.println(
            "ERROR: JPEG file is empty."
        );


        server.send(
            400,
            "application/json",
            "{\"error\":\"Empty JPEG received\"}"
        );


        return;
    }


    Serial.println();

    Serial.println(
        "================================"
    );

    Serial.println(
        "Processing JPEG from SPIFFS..."
    );

    Serial.println(
        "================================"
    );


    Serial.printf(
        "JPEG size: %u bytes\n",
        (unsigned int)jpeg_size
    );


    Serial.printf(
        "Free heap before decode: %u bytes\n",
        ESP.getFreeHeap()
    );


    // ========================================================
    // GET JPEG DIMENSIONS
    // ========================================================

    uint16_t jpg_width = 0;

    uint16_t jpg_height = 0;


    JRESULT size_result =
        TJpgDec.getFsJpgSize(
            &jpg_width,
            &jpg_height,
            JPEG_PATH
        );


    if (
        size_result != JDR_OK
    )
    {
        Serial.printf(
            "ERROR: Invalid JPEG. Code=%d\n",
            size_result
        );


        server.send(
            400,
            "application/json",
            "{\"error\":\"Invalid JPEG image\"}"
        );


        return;
    }


    original_width =
        jpg_width;


    original_height =
        jpg_height;


    Serial.printf(
        "Original JPEG: %d x %d\n",
        original_width,
        original_height
    );


    // ========================================================
    // JPEG SCALE
    // ========================================================

    TJpgDec.setJpgScale(4);


    decoded_width =
        (original_width + 3) / 4;


    decoded_height =
        (original_height + 3) / 4;


    Serial.printf(
        "Decoded image: %d x %d\n",
        decoded_width,
        decoded_height
    );


    // ========================================================
    // CLEAR IMAGE BUFFER
    // ========================================================

    memset(
        image_buffer,
        0,
        sizeof(image_buffer)
    );


    // ========================================================
    // DECODE JPEG
    // ========================================================

    TJpgDec.setCallback(
        jpegCallback
    );


    unsigned long decode_start =
        millis();


    JRESULT decode_result =
        TJpgDec.drawFsJpg(
            0,
            0,
            JPEG_PATH
        );


    unsigned long decode_time =
        millis() - decode_start;


    if (
        decode_result != JDR_OK
    )
    {
        Serial.printf(
            "ERROR: JPEG decode failed. Code=%d\n",
            decode_result
        );


        server.send(
            400,
            "application/json",
            "{\"error\":\"JPEG decoding failed. Please use a standard JPEG image.\"}"
        );


        return;
    }


    Serial.println(
        "JPEG decoding successful."
    );


    Serial.printf(
        "JPEG decode time: %lu ms\n",
        decode_time
    );


    // ========================================================
    // RUN FOMO
    // ========================================================

    float max_probability =
        0.0f;


    int best_x = 0;

    int best_y = 0;


    unsigned long inference_time =
        0;


    bool inference_ok =
        runInference(
            max_probability,
            best_x,
            best_y,
            inference_time
        );


    if (!inference_ok)
    {
        server.send(
            500,
            "application/json",
            "{\"error\":\"AI inference failed\"}"
        );


        return;
    }


    // ========================================================
    // DECISION
    // ========================================================

    bool detected =
        max_probability >=
        SCAR_THRESHOLD;


    // ========================================================
    // SERIAL RESULT
    // ========================================================

    Serial.println();

    Serial.println(
        "========== RESULT =========="
    );


    Serial.printf(
        "Maximum probability: %.4f\n",
        max_probability
    );


    Serial.printf(
        "Confidence: %.2f%%\n",
        max_probability * 100.0f
    );


    Serial.printf(
        "Strongest cell: (%d,%d)\n",
        best_x,
        best_y
    );


    Serial.printf(
        "Inference time: %lu ms\n",
        inference_time
    );


    if (detected)
    {
        Serial.println(
            "SCAR DETECTED / TRACK ABNORMAL"
        );
    }
    else
    {
        Serial.println(
            "NORMAL TRACK"
        );
    }


    Serial.println(
        "============================"
    );


    // ========================================================
    // JSON RESPONSE
    // ========================================================

    String json = "{";


    // Detection

    json +=
        "\"detected\":";


    json +=
        detected
        ? "true"
        : "false";


    // Confidence

    json +=
        ",\"confidence\":";


    json +=
        String(
            max_probability,
            6
        );


    // Strongest X

    json +=
        ",\"cell_x\":";


    json +=
        String(
            best_x
        );


    // Strongest Y

    json +=
        ",\"cell_y\":";


    json +=
        String(
            best_y
        );


    // Inference time

    json +=
        ",\"inference_ms\":";


    json +=
        String(
            inference_time
        );


    // Decode time

    json +=
        ",\"decode_ms\":";


    json +=
        String(
            decode_time
        );


    // Image width

    json +=
        ",\"image_width\":";


    json +=
        String(
            original_width
        );


    // Image height

    json +=
        ",\"image_height\":";


    json +=
        String(
            original_height
        );


    // ========================================================
    // FOMO GRID
    // ========================================================

    json +=
        ",\"fomo_grid\":[";


    for (
        int i = 0;
        i < FOMO_GRID_CELLS;
        i++
    )
    {
        if (i > 0)
        {
            json += ",";
        }


        json +=
            String(
                fomo_grid[i],
                4
            );
    }


    json +=
        "]";


    // End JSON

    json += "}";


    // ========================================================
    // SEND RESULT
    // ========================================================

    server.send(
        200,
        "application/json",
        json
    );
}


// ============================================================
// ROOT PAGE
// ============================================================

void handleRoot()
{
    server.send_P(
        200,
        "text/html",
        DASHBOARD_HTML
    );
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(
        115200
    );


    delay(
        1000
    );


    Serial.println();

    Serial.println(
        "===================================="
    );

    Serial.println(
        "       Railway FOMO ESP32"
    );

    Serial.println(
        "     Edge AI Scar Detection"
    );

    Serial.println(
        "===================================="
    );


    Serial.printf(
        "Free heap at startup: %u bytes\n",
        ESP.getFreeHeap()
    );


    // ========================================================
    // SPIFFS
    // ========================================================

    Serial.println();

    Serial.println(
        "Initializing SPIFFS..."
    );


    if (
        !SPIFFS.begin(true)
    )
    {
        Serial.println(
            "ERROR: SPIFFS initialization failed."
        );


        while (true)
        {
            delay(1000);
        }
    }


    Serial.println(
        "SPIFFS initialized."
    );


    Serial.printf(
        "SPIFFS total: %u bytes\n",
        SPIFFS.totalBytes()
    );


    Serial.printf(
        "SPIFFS used: %u bytes\n",
        SPIFFS.usedBytes()
    );


    Serial.printf(
        "SPIFFS free: %u bytes\n",
        SPIFFS.totalBytes() -
        SPIFFS.usedBytes()
    );


    // ========================================================
    // MODEL
    // ========================================================

    if (
        !initializeModel()
    )
    {
        Serial.println();

        Serial.println(
            "MODEL INITIALIZATION FAILED."
        );


        while (true)
        {
            delay(1000);
        }
    }


    Serial.printf(
        "Free heap after model: %u bytes\n",
        ESP.getFreeHeap()
    );


    // ========================================================
    // WIFI ACCESS POINT
    // ========================================================

    WiFi.mode(
        WIFI_AP
    );


    bool wifi_ok =
        WiFi.softAP(
            AP_SSID,
            AP_PASSWORD
        );


    if (!wifi_ok)
    {
        Serial.println(
            "ERROR: Wi-Fi AP failed."
        );


        while (true)
        {
            delay(1000);
        }
    }


    IPAddress ip =
        WiFi.softAPIP();


    Serial.println();

    Serial.println(
        "Wi-Fi Access Point started."
    );


    Serial.print(
        "SSID: "
    );

    Serial.println(
        AP_SSID
    );


    Serial.print(
        "Password: "
    );

    Serial.println(
        AP_PASSWORD
    );


    Serial.print(
        "IP Address: "
    );

    Serial.println(
        ip
    );


    // ========================================================
    // JPEG DECODER
    // ========================================================

    TJpgDec.setCallback(
        jpegCallback
    );


    TJpgDec.setJpgScale(4);


    // ========================================================
    // WEB SERVER
    // ========================================================

    server.on(
        "/",
        HTTP_GET,
        handleRoot
    );


    server.on(
        "/upload",
        HTTP_POST,
        handleUploadComplete,
        handleUploadData
    );


    server.begin();


    Serial.println();

    Serial.println(
        "Web server started."
    );


    Serial.println(
        "Open: http://192.168.4.1"
    );


    Serial.println();

    Serial.println(
        "===================================="
    );


    Serial.println(
        "          SYSTEM READY"
    );


    Serial.println(
        "===================================="
    );
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
    server.handleClient();
}