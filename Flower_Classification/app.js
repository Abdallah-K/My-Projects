


$(function(){
    $('#file').change(function(e){
        var x = URL.createObjectURL(e.target.files[0]);
        $('#image').attr("src",x)
    })

    var api = 'http://127.0.0.1:8000/flowers'

    $("#upload-form").submit((e)=>{
        e.preventDefault();
        let formdata = new FormData();
        let file = $("#file")[0].files[0];
        formdata.append('file',file);

        $.ajax({
            url:api,
            type:'post',
            data:formdata,
            contentType:false,
            processData:false,
            success:(data)=>{
                console.log(data);
                $('.pretext').html(`<span class="iconify" data-icon="emojione-v1:bouquet-of-flowers"></span>${data.Prediction}`);
                $('.overtext').html(data.Overview);
            }
        })
    })
})