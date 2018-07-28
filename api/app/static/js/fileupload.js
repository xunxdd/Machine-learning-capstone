$(function () {
    'use strict';
    var url = '/api/upload';
    var textObj = {
        'Normal': 'Great, this is a normal chest X-ray',
        'Pneumonia - Bacterial': 'It appears to be bacterial pneumonia. You will need antibiotics. ',
        'Pneumonia - Viral': 'It appears to be viral pneumonia. In general, viral pneumonia needs no medication.',
    };
    $('#fileupload').fileupload({
        url: url,
        dataType: 'json',
        done: function (e, data) {
            console.log(data.result);
            var imgUrl = '/static/uploadimgs/' + data.result.file;
            $('#img-uploaded').attr('src', imgUrl);
            if (textObj[data.result.label]) {
                $('#result').text(textObj[data.result.label]);
            } else {
                $('#result').text('Sorry, something went wrong. I am not able to tell what this image is');
            }
        },
        progressall: function (e, data) {
            var progress = parseInt(data.loaded / data.total * 100, 10);
            $('#progress .progress-bar').css(
                'width',
                progress + '%'
            );
        }
    }).prop('disabled', !$.support.fileInput)
        .parent().addClass($.support.fileInput ? undefined : 'disabled');
});