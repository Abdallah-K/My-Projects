<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.1/jquery.min.js" integrity="sha512-aVKKRRi/Q/YV+4mjoKBsE4x3H+BkegoM/em46NNlCqNTmUYADjBbeNefNxYV7giUp0VxICtqdrbqU7iVaeZNXA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css" integrity="sha512-xh6O/CkQoPOWDdYTDqeRdPCVd1SpvCA9XXcUnZS2FmJNp1coAFzvtCN9BmamE+4aHK8yyUHUSCcJHgXloTyT2A==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <link rel="stylesheet" href="./views/styles/quotepage.css"/>
    <link rel="shortcut icon" type="image/png" href="./views/images/logo.png"/>
    <title>Anime World</title>
</head>
<body>
    <?php include("views/components/navbar.php")?>
    <section id="QuoPage">
        <div class="quotebox">
            <div class="quotetitle">
                <h1>Famouse Anime Quotes</h1>
            </div>
            <div class="quotebdy">


            </div>
        </div>
    </section>


</body>
</html>



<script>

    $(function(){
        
        var quotecard = "";
        $.post("API/PostModel.php",{
            action:"select",
            tablename:"quotesdata"
        },function(data){
            data = $.parseJSON(data);
            $.each(data,(index,quoteitem)=>{
                quotecard += `
                    <div class="quoteCard">
                        <div class="quotimg">
                            <img src="./views/images/quotes/${quoteitem.image}">
                        </div>
                        <div class="quoteinfo">
                            <div class="quoteinfotitle">
                                <h1>${quoteitem.name}</h1>
                            </div>
                            <div class="quoteinfodata">
                                <p>${quoteitem.quote}</p>
                            </div>
                        </div>
                    </div>`;
            })
            $(".quotebdy").html(quotecard);
        })








        // $.ajax({
        //     url:"Models/selectQuote.php",
        //     method:"post",
        //     success:function(response){
        //         response = $.parseJSON(response);
        //         $.each(response,(index,quotcard)=>{
        //             outcard +=`
        //             <div class="quoteCard">
        //                 <div class="quotimg">
        //                     <img src="./views/images/quotes/${quotcard.image}">
        //                 </div>
        //                 <div class="quoteinfo">
        //                     <div class="quoteinfotitle">
        //                         <h1>${quotcard.name}</h1>
        //                     </div>
        //                     <div class="quoteinfodata">
        //                         <p>${quotcard.quote}</p>
        //                     </div>
        //                 </div>
        //             </div>`;
        //         })
        //         $(".quotebdy").html(outcard);
        //     }
        // })
    })




</script>