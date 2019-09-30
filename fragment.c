#version 450

in flat vec4 ex_color;

void main()
{
   gl_FragColor = ex_color;
   gl_FragColor.x = ex_color.x;
   gl_FragColor.y = ex_color.y;
   gl_FragColor.z = ex_color.z;
   gl_FragColor.w = 0.1f + ex_color.w * 0.0f;

   //float z = (2.0f * near * far) / (far + near - (gl_FragCoord.z * 2.0f - 1.0f) * (far - near)) / far;
   //gl_FragColor = vec4(vec3(z), 1.0);
   gl_FragColor = vec4(0.5, 1.0, 0.0, 0.05);
}